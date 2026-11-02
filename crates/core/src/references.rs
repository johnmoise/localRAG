//! References from each document's bibliography (docs/ARCHITECTURE.md §7.1): stored per
//! document, enriched from Crossref (DOI lookup, else a bibliographic search on the citation
//! string) including the abstract, kept as stubs when Crossref has nothing, linked to the
//! corpus document they refer to when it's in `corpus/`, and each embedded as its own chunk
//! (in `reference_vec`, apart from the corpus chunks) for suggesting additions.

use std::collections::{BTreeMap, HashMap, HashSet};

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;
use serde_json::{Value, json};

use crate::bibliography::Bibliography;
use crate::db;
use crate::embed::Embedder;
use crate::metadata::crossref::{Crossref, Work};
use crate::scan::sha256_str;

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
                save_work(conn, id, &w)?;
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

/// What a matched reference's stored Crossref data includes: 1 = abstract, 2 = titles
/// without markup. Older rows are re-read (from the cache) by `backfill_details`.
pub const DETAILS_VERSION: i64 = 2;

fn save_work(conn: &Connection, id: i64, w: &Work) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE doc_references SET status = 'matched', csl_json = ?2, doi = ?3, title = COALESCE(?4, title), year = COALESCE(?5, year),
           cited_by = ?6, abstract = ?7, details_version = ?9, checked_at = ?8 WHERE id = ?1",
        params![id, w.to_csl().to_string(), w.doi, w.title, w.year, w.cited_by, w.abstract_text, db::now_rfc3339(), DETAILS_VERSION],
    )?;
    Ok(())
}

/// Re-reads the full record (with abstract) for references matched before the current
/// `DETAILS_VERSION`. DOI lookups are usually answered from the cache. Stops at the first network error.
pub fn backfill_details(conn: &Connection, client: &Crossref, progress: &mut dyn FnMut(usize, usize)) -> anyhow::Result<usize> {
    let todo: Vec<(i64, String)> = conn
        .prepare("SELECT id, doi FROM doc_references WHERE status = 'matched' AND details_version < ?1 AND doi IS NOT NULL")?
        .query_map([DETAILS_VERSION], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    let total = todo.len();
    for (i, (id, doi)) in todo.into_iter().enumerate() {
        progress(i, total);
        match client.work(&doi, false)? {
            Some(w) => save_work(conn, id, &w)?,
            None => {
                conn.execute("UPDATE doc_references SET details_version = ?2 WHERE id = ?1", [id, DETAILS_VERSION])?;
            }
        }
    }
    progress(total, total);
    Ok(total)
}

/// The text a reference is embedded as: its enriched citation (title, authors, year, venue)
/// and abstract, or for a stub the citation string itself.
pub fn chunk_text(title: Option<&str>, raw: &str, csl: Option<&str>, abstract_text: Option<&str>) -> String {
    const MAX_CHARS: usize = 6000;
    let v = csl.and_then(|c| serde_json::from_str::<Value>(c).ok());
    let mut out = match (&v, title) {
        (Some(v), Some(t)) => {
            let authors: Vec<String> = v["author"]
                .as_array()
                .into_iter()
                .flatten()
                .take(6)
                .filter_map(|a| match (a["given"].as_str(), a["family"].as_str(), a["literal"].as_str()) {
                    (Some(g), Some(f), _) => Some(format!("{g} {f}")),
                    (None, Some(f), _) => Some(f.to_string()),
                    (_, _, Some(l)) => Some(l.to_string()),
                    _ => None,
                })
                .collect();
            let year = v["issued"]["date-parts"][0][0].as_i64().map(|y| format!(" ({y})")).unwrap_or_default();
            let venue = v["container-title"].as_str().map(|c| format!(" {c}.")).unwrap_or_default();
            format!("{t}\n{}{year}.{venue}", authors.join(", "))
        }
        _ => raw.to_string(),
    };
    if let Some(a) = abstract_text.filter(|a| !a.trim().is_empty()) {
        out.push_str("\n\n");
        out.push_str(a.trim());
    }
    if out.chars().count() > MAX_CHARS {
        out = out.chars().take(MAX_CHARS).collect();
    }
    out
}

#[derive(Debug, Default, Serialize)]
pub struct EmbedReport {
    pub embedded: usize,
    pub reused: usize,
}

/// Embeds every looked-up reference whose chunk text changed since it was last embedded
/// (new, enriched, or re-parsed). Identical texts (the same work cited by several documents)
/// are embedded once.
pub fn embed_pending(conn: &Connection, embedder: &dyn Embedder) -> anyhow::Result<EmbedReport> {
    let rows: Vec<ChunkRow> = conn
        .prepare(
            "SELECT r.id, r.title, r.raw, r.csl_json, r.abstract, r.embed_text_sha FROM doc_references r JOIN documents d ON d.id = r.doc_id
             WHERE r.status IN ('matched', 'stub') AND d.status IN ('indexed', 'needs_ocr')",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)))?
        .collect::<Result<_, _>>()?;
    let mut todo: Vec<(i64, String, String)> = Vec::new();
    for (id, title, raw, csl, abs, done) in rows {
        let status_title = if csl.is_some() { title.as_deref() } else { None };
        let text = chunk_text(status_title, &raw, csl.as_deref(), abs.as_deref());
        let sha = sha256_str(&text);
        if done.as_deref() != Some(sha.as_str()) {
            todo.push((id, text, sha));
        }
    }
    let mut report = EmbedReport::default();
    if todo.is_empty() {
        return Ok(report);
    }
    // Vectors already stored for the same text (another document citing the same work).
    let mut known: HashMap<String, Vec<u8>> = HashMap::new();
    if db::has_table(conn, "reference_vec")? {
        let mut q = conn.prepare("SELECT v.embedding FROM doc_references r JOIN reference_vec v ON v.rowid = r.id WHERE r.embed_text_sha = ?1 LIMIT 1")?;
        for (_, _, sha) in &todo {
            if !known.contains_key(sha)
                && let Some(b) = q.query_row([sha], |r| r.get::<_, Vec<u8>>(0)).optional()?
            {
                known.insert(sha.clone(), b);
            }
        }
    }
    let mut texts: Vec<String> = Vec::new();
    let mut shas: Vec<String> = Vec::new();
    for (_, text, sha) in &todo {
        if !known.contains_key(sha) && !shas.contains(sha) {
            shas.push(sha.clone());
            texts.push(text.clone());
        }
    }
    if !texts.is_empty() {
        let vectors = embedder.embed_documents(&texts)?;
        db::ensure_reference_vec(conn, vectors[0].len())?;
        for (sha, v) in shas.iter().zip(&vectors) {
            known.insert(sha.clone(), db::vec_to_blob(v));
        }
        report.embedded = texts.len();
    }
    let tx = conn.unchecked_transaction()?;
    for (id, _, sha) in &todo {
        tx.execute("DELETE FROM reference_vec WHERE rowid = ?1", [id])?;
        tx.execute("INSERT INTO reference_vec(rowid, embedding) VALUES (?1, ?2)", params![id, known[sha]])?;
        tx.execute("UPDATE doc_references SET embed_text_sha = ?2 WHERE id = ?1", params![id, sha])?;
    }
    tx.commit()?;
    report.reused = todo.len() - report.embedded.min(todo.len());
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
/// (id, title, raw, csl, abstract, embedded text sha) of a reference to embed.
type ChunkRow = (i64, Option<String>, String, Option<String>, Option<String>, Option<String>);
/// (id, citing doc, doi, title, year, raw) of a reference.
type RefKey = (i64, i64, Option<String>, Option<String>, Option<i64>, String);
/// (id, doi, title, year, raw, csl, cited_by, abstract, citing label) of a reference not in corpus/.
type MissingRow = (i64, Option<String>, Option<String>, Option<i64>, String, Option<String>, Option<i64>, Option<String>, Option<String>);

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
    /// From Crossref, when the publisher deposited one.
    #[serde(default, rename = "abstract")]
    pub abstract_text: Option<String>,
    /// Cosine similarity of its embedded reference chunk to the question (None: ranked by
    /// shared words because no reference embeddings exist yet).
    #[serde(default)]
    pub similarity: Option<f32>,
}

/// The same, as stored with an answer.
pub type MissingReferenceOwned = MissingReference;

/// Reference chunks less similar to the question than this aren't offered to the model.
pub const MIN_REFERENCE_SIMILARITY: f32 = 0.35;

/// References not in corpus/ that are closest to `question`, best first, deduplicated across
/// citing documents (by DOI, else normalized title or citation). Ranked by the similarity of
/// each reference's embedded chunk (citation + abstract) to the question; when nothing is
/// embedded yet (or no embedder is given), by words shared with the title and venue.
pub fn missing_for_question(conn: &Connection, embedder: Option<&dyn Embedder>, question: &str, n: usize) -> anyhow::Result<Vec<MissingReference>> {
    let has_vectors = db::has_table(conn, "reference_vec")? && conn.query_row("SELECT count(*) FROM reference_vec", [], |r| r.get::<_, i64>(0))? > 0;
    let scores: HashMap<i64, f32> = match embedder {
        Some(e) if has_vectors => {
            let q = e.embed_query(question)?;
            conn.prepare("SELECT rowid, distance FROM reference_vec WHERE embedding MATCH ?1 AND k = 400")?
                .query_map([db::vec_to_blob(&q)], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, f64>(1)?)))?
                .map(|r| r.map(|(id, d)| (id, (1.0 - d * d / 2.0) as f32)))
                .filter(|r| r.as_ref().map_or(true, |(_, s)| *s >= MIN_REFERENCE_SIMILARITY))
                .collect::<Result<_, _>>()?
        }
        _ => return missing_lexical(conn, question, n),
    };
    let mut ranked: Vec<(MissingReference, f32)> = group(load_missing(conn)?.into_iter().filter_map(|row| scores.get(&row.0).map(|s| (row, *s))));
    for (r, s) in &mut ranked {
        r.similarity = Some(*s);
    }
    // Closest first; works cited by more corpus documents break ties.
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cited_in.len().cmp(&a.0.cited_in.len())));
    Ok(ranked.into_iter().take(n).map(|(r, _)| r).collect())
}

fn load_missing(conn: &Connection) -> anyhow::Result<Vec<MissingRow>> {
    Ok(conn
        .prepare(
            "SELECT r.id, lower(r.doi), r.title, r.year, r.raw, r.csl_json, r.cited_by, r.abstract, d.cite_label
             FROM doc_references r JOIN documents d ON d.id = r.doc_id
             WHERE r.in_corpus_doc_id IS NULL AND d.status IN ('indexed', 'needs_ocr')",
        )?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?)))?
        .collect::<Result<_, _>>()?)
}

fn container_of(csl: Option<&str>) -> Option<String> {
    csl.and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|v| v["container-title"].as_str().map(str::to_string))
}

/// One entry per work (by DOI, else normalized title or citation), keeping its best score and
/// every corpus document that cites it.
fn group(rows: impl Iterator<Item = (MissingRow, f32)>) -> Vec<(MissingReference, f32)> {
    let mut groups: BTreeMap<String, (MissingReference, f32)> = BTreeMap::new();
    for ((_, doi, title, year, raw, csl, cited_by, abstract_text, citing), score) in rows {
        let key = doi.clone().or_else(|| title.as_deref().map(normalized)).unwrap_or_else(|| normalized(&raw));
        let entry = groups.entry(key).or_insert_with(|| {
            (
                MissingReference {
                    label: label(csl.as_deref(), &raw, year),
                    title: title.clone(),
                    year,
                    doi: doi.clone(),
                    container: container_of(csl.as_deref()),
                    cited_in: Vec::new(),
                    cited_by,
                    raw: raw.clone(),
                    abstract_text: abstract_text.clone(),
                    similarity: None,
                },
                score,
            )
        });
        entry.1 = entry.1.max(score);
        if let Some(c) = citing
            && !entry.0.cited_in.contains(&c)
        {
            entry.0.cited_in.push(c);
        }
    }
    groups.into_values().collect()
}

/// The fallback ranking: share of the question's content words found in title and venue.
fn missing_lexical(conn: &Connection, question: &str, n: usize) -> anyhow::Result<Vec<MissingReference>> {
    let stop: HashSet<&str> = ["what", "which", "how", "does", "the", "and", "for", "are", "with", "from", "that", "this", "into", "between", "about", "there", "their", "were", "was", "many", "much", "after", "before"].into_iter().collect();
    let q: Vec<String> = tokens(question).into_iter().filter(|w| w.chars().count() > 3 && !stop.contains(w.as_str())).collect();
    if q.is_empty() {
        return Ok(Vec::new());
    }
    let scored = load_missing(conn)?.into_iter().filter_map(|row| {
        let text = format!("{} {}", row.2.clone().unwrap_or_else(|| row.4.clone()), container_of(row.5.as_deref()).unwrap_or_default()).to_lowercase();
        let hits = q.iter().filter(|w| text.contains(w.as_str())).count();
        (hits > 0).then(|| (row, hits as f32 / q.len() as f32))
    });
    let mut all = group(scored);
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
