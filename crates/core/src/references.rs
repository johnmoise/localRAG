//! References from each document's bibliography (docs/ARCHITECTURE.md §5, §7): stored per
//! document, enriched from Crossref (DOI lookup, else a bibliographic search on the citation
//! string), kept as stubs when Crossref has nothing, and linked to the corpus document they
//! refer to when it's in `corpus/`.

use std::collections::{BTreeMap, HashSet};

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::{Value, json};

use crate::bibliography::Bibliography;
use crate::db;
use crate::metadata::crossref::{Crossref, Work};

/// Replaces a document's references with those of its (re)parsed bibliography.
/// `None` means no bibliography was found: `reference_count` becomes NULL ("N/A").
pub fn store(conn: &Connection, doc_id: i64, bib: Option<&Bibliography>) -> anyhow::Result<()> {
    let tx = conn.unchecked_transaction()?;
    tx.execute("DELETE FROM doc_references WHERE doc_id = ?1", [doc_id])?;
    match bib {
        Some(b) => {
            let mut ins = tx.prepare(
                "INSERT INTO doc_references (doc_id, ord, number, raw, doi, year, title) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            )?;
            for e in &b.entries {
                ins.execute(params![doc_id, e.ord as i64, e.number, e.text, e.doi, e.year, guess_title(&e.text)])?;
            }
            tx.execute("UPDATE documents SET reference_count = ?2 WHERE id = ?1", params![doc_id, b.entries.len() as i64])?;
        }
        None => {
            tx.execute("UPDATE documents SET reference_count = NULL WHERE id = ?1", [doc_id])?;
        }
    }
    tx.commit()?;
    Ok(())
}

#[derive(Debug, Default, Serialize)]
pub struct EnrichReport {
    pub matched: usize,
    pub stubs: usize,
    pub errors: Vec<String>,
}

/// Looks up pending references on Crossref. A reference with a DOI is fetched directly;
/// otherwise its citation string is searched and the best candidate accepted only if its
/// title, year and first author appear in the string. Everything else becomes a stub built
/// from the citation alone. Stops at the first network error (the rest stay pending).
pub fn enrich_pending(conn: &Connection, client: &Crossref, progress: &mut dyn FnMut(usize, usize)) -> anyhow::Result<EnrichReport> {
    let pending: Vec<(i64, String, Option<String>)> = conn
        .prepare(
            "SELECT r.id, r.raw, r.doi FROM doc_references r JOIN documents d ON d.id = r.doc_id
             WHERE r.status = 'pending' AND d.status IN ('indexed', 'needs_ocr') ORDER BY r.doc_id, r.ord",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    let mut report = EnrichReport::default();
    let total = pending.len();
    for (i, (id, raw, doi)) in pending.into_iter().enumerate() {
        progress(i, total);
        let found = (|| -> anyhow::Result<Option<Work>> {
            if let Some(d) = &doi
                && let Some(w) = client.work(d, false)?
            {
                return Ok(Some(w));
            }
            let candidates = client.search(&raw, 3, false)?;
            Ok(candidates.into_iter().map(|w| (score(&w, &raw), w)).filter(|(s, _)| *s >= 0.8).max_by(|a, b| a.0.total_cmp(&b.0)).map(|(_, w)| w))
        })();
        match found {
            Ok(Some(w)) => {
                conn.execute(
                    "UPDATE doc_references SET status = 'matched', csl_json = ?2, doi = ?3, title = COALESCE(?4, title), year = COALESCE(?5, year),
                       cited_by = ?6, checked_at = ?7 WHERE id = ?1",
                    params![id, w.to_csl().to_string(), w.doi, w.title, w.year, w.cited_by, db::now_rfc3339()],
                )?;
                report.matched += 1;
            }
            Ok(None) => {
                conn.execute("UPDATE doc_references SET status = 'stub', checked_at = ?2 WHERE id = ?1", params![id, db::now_rfc3339()])?;
                report.stubs += 1;
            }
            Err(e) => {
                report.errors.push(format!("{e:#}"));
                break;
            }
        }
    }
    progress(total, total);
    link(conn)?;
    Ok(report)
}

/// How well a Crossref candidate fits a raw citation string: most of its title words, plus
/// its year and first author, should appear in the string.
fn score(w: &Work, raw: &str) -> f32 {
    let raw_l = raw.to_lowercase();
    let words = tokens(w.title.as_deref().unwrap_or_default());
    if words.is_empty() {
        return 0.0;
    }
    let title = words.iter().filter(|t| raw_l.contains(t.as_str())).count() as f32 / words.len() as f32;
    let year = w.year.is_some_and(|y| raw_l.contains(&y.to_string()));
    let author = w.author.first().and_then(|a| a.family.as_deref().or(a.literal.as_deref())).is_some_and(|f| raw_l.contains(&f.to_lowercase()));
    0.75 * title + 0.15 * year as u8 as f32 + 0.10 * author as u8 as f32
}

fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.chars().count() > 2).map(str::to_string).collect()
}

/// The title inside a citation string: after the author list, before the journal.
/// "Parry C, Kent EE. Cancer survivors: a booming population. Cancer Epidemiol…" → the
/// second segment; parenthesised years ("(2019).") are skipped.
pub fn guess_title(raw: &str) -> Option<String> {
    let segments: Vec<&str> = raw.split(". ").map(str::trim).collect();
    segments
        .iter()
        .skip(1)
        .find(|s| !s.starts_with('(') && s.split_whitespace().count() >= 3)
        .map(|s| s.trim_end_matches('.').to_string())
}

/// (id, doi, title, year) of a corpus document.
type DocKey = (i64, Option<String>, Option<String>, Option<i64>);
/// (id, citing doc, doi, title, year, raw) of a reference.
type RefKey = (i64, i64, Option<String>, Option<String>, Option<i64>, String);
/// (doi, title, year, raw, csl, cited_by, citing label) of a reference not in corpus/.
type MissingRow = (Option<String>, Option<String>, Option<i64>, String, Option<String>, Option<i64>, Option<String>);

/// Sets `in_corpus_doc_id` for every reference: the corpus document with the same DOI, or
/// with (nearly) the same title and year. Run after syncs and metadata changes.
pub fn link(conn: &Connection) -> anyhow::Result<()> {
    let docs: Vec<DocKey> = conn
        .prepare(
            "SELECT id, lower(doi), json_extract(csl_json, '$.title'), json_extract(csl_json, '$.issued.\"date-parts\"[0][0]')
             FROM documents WHERE status != 'removed'",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))?
        .collect::<Result<_, _>>()?;
    let refs: Vec<RefKey> = conn
        .prepare("SELECT id, doc_id, lower(doi), title, year, raw FROM doc_references")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))?
        .collect::<Result<_, _>>()?;
    let tx = conn.unchecked_transaction()?;
    for (id, citing, doi, title, year, raw) in refs {
        let raw_l = raw.to_lowercase();
        let hit = docs.iter().filter(|d| d.0 != citing).find(|(_, ddoi, dtitle, dyear)| {
            if let (Some(a), Some(b)) = (&doi, ddoi) {
                return a == b;
            }
            let Some(dt) = dtitle.as_deref() else { return false };
            let years_ok = match (year, dyear) {
                (Some(a), Some(b)) => a == *b,
                _ => true,
            };
            let by_title = title.as_deref().is_some_and(|t| crate::metadata::title_similarity(t, dt) >= 0.85);
            // Stubs: the corpus document's title appears, word for word, in the citation.
            let in_raw = tokens(dt).len() >= 4 && normalized(&raw_l).contains(&normalized(dt));
            years_ok && (by_title || in_raw)
        });
        tx.execute("UPDATE doc_references SET in_corpus_doc_id = ?2 WHERE id = ?1", params![id, hit.map(|d| d.0)])?;
    }
    tx.commit()?;
    Ok(())
}

fn normalized(s: &str) -> String {
    tokens(s).join(" ")
}

#[derive(Debug, Clone, Serialize)]
pub struct ReferenceView {
    pub ord: i64,
    pub number: Option<i64>,
    /// "Parry et al. 2011"
    pub label: String,
    pub title: Option<String>,
    pub year: Option<i64>,
    pub doi: Option<String>,
    /// matched (Crossref) or stub (from the citation only) or pending.
    pub status: String,
    pub raw: String,
    pub cited_by: Option<i64>,
    /// The corpus document this reference is, if it's in corpus/.
    pub in_corpus: Option<Value>,
}

pub fn for_document(conn: &Connection, doc_id: i64) -> anyhow::Result<Vec<ReferenceView>> {
    let rows = conn
        .prepare(
            "SELECT r.ord, r.number, r.title, r.year, r.doi, r.status, r.raw, r.csl_json, r.cited_by, d.id, d.cite_label, d.path
             FROM doc_references r LEFT JOIN documents d ON d.id = r.in_corpus_doc_id
             WHERE r.doc_id = ?1 ORDER BY r.ord",
        )?
        .query_map([doc_id], |r| {
            let raw: String = r.get(6)?;
            let csl: Option<String> = r.get(7)?;
            let year: Option<i64> = r.get(3)?;
            let corpus_id: Option<i64> = r.get(9)?;
            Ok(ReferenceView {
                ord: r.get(0)?,
                number: r.get(1)?,
                label: label(csl.as_deref(), &raw, year),
                title: r.get(2)?,
                year,
                doi: r.get(4)?,
                status: r.get(5)?,
                raw,
                cited_by: r.get(8)?,
                in_corpus: corpus_id.map(|id| json!({ "id": id, "label": r.get::<_, Option<String>>(10).ok().flatten(), "path": r.get::<_, Option<String>>(11).ok().flatten() })),
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// "Smith et al. 2021" from Crossref data, else from the citation's first surname and year.
fn label(csl: Option<&str>, raw: &str, year: Option<i64>) -> String {
    if let Some(l) = csl.and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|v| {
        let fam = v["author"][0]["family"].as_str().or(v["author"][0]["literal"].as_str())?.to_string();
        let many = v["author"].as_array().is_some_and(|a| a.len() > 2);
        let two = v["author"].as_array().filter(|a| a.len() == 2).and_then(|a| a[1]["family"].as_str().map(str::to_string));
        let who = match (many, two) {
            (true, _) => format!("{fam} et al."),
            (false, Some(b)) => format!("{fam} & {b}"),
            _ => fam,
        };
        Some(match v["issued"]["date-parts"][0][0].as_i64().or(year) {
            Some(y) => format!("{who} {y}"),
            None => who,
        })
    }) {
        return l;
    }
    let first = raw.split([',', ' ', ';']).find(|w| w.chars().count() > 1).unwrap_or("?");
    match year {
        Some(y) => format!("{first} {y}"),
        None => first.to_string(),
    }
}

/// A work cited somewhere in the corpus but not itself in corpus/, ranked for a question.
#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub struct MissingReference {
    pub label: String,
    pub title: Option<String>,
    pub year: Option<i64>,
    pub doi: Option<String>,
    pub container: Option<String>,
    /// How many corpus documents cite it, and which.
    pub cited_in: Vec<String>,
    /// Crossref citation count, when known.
    pub cited_by: Option<i64>,
    /// The citation string (for stubs, the only information there is).
    pub raw: String,
}

/// The same, as stored with an answer.
pub type MissingReferenceOwned = MissingReference;

/// References not in corpus/ whose title (or citation) shares words with `question`, best
/// first. Deduplicated across citing documents (by DOI, else normalized title or citation).
/// Ranking is lexical: the bibliography is deliberately never embedded.
pub fn missing_for_question(conn: &Connection, question: &str, n: usize) -> anyhow::Result<Vec<MissingReference>> {
    let stop: HashSet<&str> = ["what", "which", "how", "does", "the", "and", "for", "are", "with", "from", "that", "this", "into", "between", "about", "there", "their", "were", "was", "many", "much", "after", "before"].into_iter().collect();
    let q: Vec<String> = tokens(question).into_iter().filter(|w| w.chars().count() > 3 && !stop.contains(w.as_str())).collect();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<MissingRow> = conn
        .prepare(
            "SELECT lower(r.doi), r.title, r.year, r.raw, r.csl_json, r.cited_by, d.cite_label
             FROM doc_references r JOIN documents d ON d.id = r.doc_id
             WHERE r.in_corpus_doc_id IS NULL AND d.status IN ('indexed', 'needs_ocr')",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?)))?
        .collect::<Result<_, _>>()?;
    let mut groups: BTreeMap<String, (MissingReference, f32)> = BTreeMap::new();
    for (doi, title, year, raw, csl, cited_by, citing) in rows {
        let key = doi.clone().or_else(|| title.as_deref().map(normalized)).unwrap_or_else(|| normalized(&raw));
        let container = csl.as_deref().and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|v| v["container-title"].as_str().map(str::to_string));
        let text = format!("{} {}", title.clone().unwrap_or_else(|| raw.clone()), container.clone().unwrap_or_default()).to_lowercase();
        let hits = q.iter().filter(|w| text.contains(w.as_str())).count();
        if hits == 0 {
            continue;
        }
        let score = hits as f32 / q.len() as f32;
        let entry = groups.entry(key).or_insert_with(|| {
            (
                MissingReference {
                    label: label(csl.as_deref(), &raw, year),
                    title: title.clone(),
                    year,
                    doi: doi.clone(),
                    container: container.clone(),
                    cited_in: Vec::new(),
                    cited_by,
                    raw: raw.clone(),
                },
                score,
            )
        });
        if let Some(c) = citing
            && !entry.0.cited_in.contains(&c)
        {
            entry.0.cited_in.push(c);
        }
    }
    let mut all: Vec<(MissingReference, f32)> = groups.into_values().collect();
    // Relevance first; works cited by more corpus documents break ties.
    all.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cited_in.len().cmp(&a.0.cited_in.len())));
    Ok(all.into_iter().take(n).map(|(r, _)| r).collect())
}

/// Fills in `documents.cited_by` for documents matched before it was recorded (cache hits).
pub fn backfill_cited_by(conn: &Connection, client: &Crossref) -> anyhow::Result<()> {
    let todo: Vec<(i64, String)> = conn
        .prepare("SELECT id, doi FROM documents WHERE cited_by IS NULL AND doi IS NOT NULL AND metadata_source LIKE 'crossref%'")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    for (id, doi) in todo {
        if let Some(w) = client.work(&doi, false)? {
            conn.execute("UPDATE documents SET cited_by = ?2 WHERE id = ?1", params![id, w.cited_by])?;
        }
    }
    Ok(())
}

/// Reference counts for the Documents tab: (references, of which in corpus/).
pub fn counts(conn: &Connection, doc_id: i64) -> anyhow::Result<Option<(i64, i64)>> {
    let total: Option<i64> = conn.query_row("SELECT reference_count FROM documents WHERE id = ?1", [doc_id], |r| r.get(0)).optional()?.flatten();
    let Some(total) = total else { return Ok(None) };
    let inside: i64 = conn.query_row("SELECT count(*) FROM doc_references WHERE doc_id = ?1 AND in_corpus_doc_id IS NOT NULL", [doc_id], |r| r.get(0))?;
    Ok(Some((total, inside)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn titles_from_citations() {
        assert_eq!(guess_title("Parry C, Kent EE, Rowland JH. Cancer survivors: a booming population. Cancer Epidemiol Biomarkers Prev. 2011;20:1996–2005.").as_deref(), Some("Cancer survivors: a booming population"));
        assert_eq!(guess_title("Yao, W.; Mei, C.; Hui, L. Evaluation and comparison of DNA degradation kinetics. Gene 2016, 590, 142–148.").as_deref(), Some("Evaluation and comparison of DNA degradation kinetics"));
        assert_eq!(guess_title("Smith, J. (2019). A long title about attention. Journal of Things, 3.").as_deref(), Some("A long title about attention"));
    }
}
