//! Citation metadata (ARCHITECTURE.md §7): find each document's DOI or best Crossref match,
//! store CSL-JSON, assign `[Author Year]` labels (with a/b for collisions), and derive printed
//! page numbers from the article's page range.

pub mod crossref;
pub mod ratelimit;

use anyhow::Context;
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::db;
use crossref::{Crossref, Work};

/// What happened to one document's metadata.
#[derive(Debug, Clone, Serialize)]
pub struct MetadataEvent {
    pub doc_id: i64,
    pub path: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Default, Serialize)]
pub struct MetadataReport {
    pub matched: usize,
    pub proposed: usize,
    pub unmatched: usize,
    /// Network or Crossref errors: those documents stay pending for the next run.
    pub errors: Vec<String>,
}

struct Pending {
    id: i64,
    path: String,
    kind: String,
    doi: Option<String>,
    csl: Option<String>,
}

/// Looks up every document whose metadata is `pending`. Stops at the first network error,
/// leaving the rest pending (offline shouldn't spam retries).
pub fn process_pending(conn: &Connection, client: &Crossref, on: &mut dyn FnMut(&MetadataEvent)) -> anyhow::Result<MetadataReport> {
    let pending: Vec<Pending> = conn
        .prepare(
            "SELECT id, path, kind, doi, csl_json FROM documents
             WHERE metadata_status = 'pending' AND status IN ('indexed', 'needs_ocr') ORDER BY id",
        )?
        .query_map([], |r| Ok(Pending { id: r.get(0)?, path: r.get(1)?, kind: r.get(2)?, doi: r.get(3)?, csl: r.get(4)? }))?
        .collect::<Result<_, _>>()?;
    let mut report = MetadataReport::default();
    for doc in pending {
        match lookup(conn, client, &doc, false) {
            Ok(event) => {
                match event.status.as_str() {
                    "matched" => report.matched += 1,
                    "proposed" => report.proposed += 1,
                    _ => report.unmatched += 1,
                }
                on(&event);
            }
            Err(e) => {
                conn.execute("UPDATE documents SET metadata_error = ?2 WHERE id = ?1", params![doc.id, format!("{e:#}")])?;
                report.errors.push(format!("{}: {e:#}", doc.path));
                break;
            }
        }
    }
    assign_labels(conn)?;
    Ok(report)
}

/// Re-queries Crossref for one document, bypassing the cache (Documents → Refresh metadata).
pub fn refresh(conn: &Connection, client: &Crossref, doc_id: i64) -> anyhow::Result<MetadataEvent> {
    let doc: Pending = conn
        .query_row("SELECT id, path, kind, doi, csl_json FROM documents WHERE id = ?1", [doc_id], |r| {
            Ok(Pending { id: r.get(0)?, path: r.get(1)?, kind: r.get(2)?, doi: r.get(3)?, csl: r.get(4)? })
        })
        .with_context(|| format!("no document {doc_id}"))?;
    let event = lookup(conn, client, &doc, true)?;
    assign_labels(conn)?;
    Ok(event)
}

fn lookup(conn: &Connection, client: &Crossref, doc: &Pending, refresh: bool) -> anyhow::Result<MetadataEvent> {
    let text = leading_text(conn, doc.id)?;
    let doi = doc.doi.clone().or_else(|| find_doi(&text));
    let event = |status: &str, detail: String| MetadataEvent { doc_id: doc.id, path: doc.path.clone(), status: status.into(), detail };

    if let Some(doi) = &doi
        && let Some(work) = client.work(doi, refresh)?
    {
        apply_match(conn, doc.id, &work, "crossref_doi")?;
        return Ok(event("matched", format!("DOI {}", work.doi)));
    }

    // No DOI (or Crossref doesn't know it): bibliographic query from what we have.
    let local: Option<Value> = doc.csl.as_deref().and_then(|c| serde_json::from_str(c).ok());
    let title = local
        .as_ref()
        .and_then(|v| v["title"].as_str().map(str::to_string))
        .filter(|t| plausible_title(t))
        .or_else(|| title_from_filename(&doc.path));
    let query = match &title {
        Some(t) => format!("{t} {}", text.chars().take(300).collect::<String>()),
        None => text.chars().take(500).collect(),
    };
    if query.trim().is_empty() || doc.kind != "pdf" && title.is_none() {
        set_status(conn, doc.id, "unmatched", None)?;
        return Ok(event("unmatched", "no DOI and too little text to search Crossref".into()));
    }
    let candidates = client.search(&query, 5, refresh)?;
    let scored: Vec<(f32, &Work)> = candidates.iter().map(|w| (score(w, title.as_deref(), &text), w)).collect();
    let best = scored.iter().cloned().max_by(|a, b| a.0.total_cmp(&b.0));
    match best {
        Some((s, w)) if s >= 0.85 => {
            apply_match(conn, doc.id, w, "crossref_query")?;
            Ok(event("matched", format!("by title ({:.0}% match): {}", s * 100.0, w.title.clone().unwrap_or_default())))
        }
        Some((s, _)) if s >= 0.45 => {
            let list: Vec<Value> = scored
                .iter()
                .filter(|(sc, _)| *sc >= 0.3)
                .map(|(sc, w)| json!({ "score": sc, "work": w }))
                .collect();
            set_status(conn, doc.id, "proposed", Some(&Value::Array(list).to_string()))?;
            Ok(event("proposed", format!("possible match ({:.0}%); confirm it on the Documents tab", s * 100.0)))
        }
        _ => {
            set_status(conn, doc.id, "unmatched", None)?;
            Ok(event("unmatched", "no confident Crossref match".into()))
        }
    }
}

/// How well a Crossref candidate fits: title similarity, plus whether its first author and
/// year appear in the document's opening text.
fn score(w: &Work, title: Option<&str>, text: &str) -> f32 {
    let Some(cand) = w.title.as_deref() else { return 0.0 };
    let text_l = text.to_lowercase();
    let title_sim = match title {
        Some(t) => dice(t, cand),
        // Without a known title, how much of the candidate's title appears in the opening text.
        None => {
            let words = tokens(cand);
            if words.is_empty() { 0.0 } else { words.iter().filter(|w| text_l.contains(w.as_str())).count() as f32 / words.len() as f32 }
        }
    };
    let author = w.author.first().and_then(|a| a.family.as_deref().or(a.literal.as_deref())).is_some_and(|f| text_l.contains(&f.to_lowercase()));
    let year = w.year.is_some_and(|y| text_l.contains(&y.to_string()));
    0.7 * title_sim + 0.15 * author as u8 as f32 + 0.15 * year as u8 as f32
}

fn tokens(s: &str) -> Vec<String> {
    s.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 2).map(str::to_string).collect()
}

/// Word-set similarity of two titles (Dice coefficient), 0–1.
pub fn title_similarity(a: &str, b: &str) -> f32 {
    dice(a, b)
}

/// Dice coefficient over word sets.
fn dice(a: &str, b: &str) -> f32 {
    let (a, b): (std::collections::HashSet<String>, std::collections::HashSet<String>) = (tokens(a).into_iter().collect(), tokens(b).into_iter().collect());
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    2.0 * a.intersection(&b).count() as f32 / (a.len() + b.len()) as f32
}

/// PDF metadata titles are often junk ("Microsoft Word - draft3.docx", "untitled").
fn plausible_title(t: &str) -> bool {
    let l = t.to_lowercase();
    t.split_whitespace().count() >= 3 && !["microsoft word", ".doc", ".pdf", "untitled", ".tex", "manuscript"].iter().any(|j| l.contains(j))
}

/// Zotero-style "Author et al. - 2021 - Title.pdf" → "Title".
fn title_from_filename(path: &str) -> Option<String> {
    let file = path.rsplit('/').next()?;
    let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
    let parts: Vec<&str> = stem.split(" - ").collect();
    (parts.len() >= 3).then(|| parts[2..].join(" - ")).filter(|t| t.split_whitespace().count() >= 3)
}

/// The first DOI-looking string: `10.<4-9 digits>/<non-space>`, minus trailing punctuation.
pub fn find_doi(text: &str) -> Option<String> {
    let lower = text.to_lowercase();
    let mut from = 0;
    while let Some(i) = lower[from..].find("10.") {
        let start = from + i;
        let rest = &lower[start + 3..];
        let digits = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if (4..=9).contains(&digits) && rest[digits..].starts_with('/') {
            // PDF line wrapping can split a DOI right after the slash ("10.3390/ jcm1368").
            let tail = rest[digits + 1..].trim_start_matches([' ', '\n']);
            let suffix: String = tail.chars().take_while(|c| !c.is_whitespace() && !"\"<>{}|\\^`".contains(*c)).collect();
            let suffix = suffix.trim_end_matches(['.', ',', ';', ')', ']', '\'']);
            if !suffix.is_empty() {
                return Some(format!("10.{}/{}", &rest[..digits], suffix));
            }
        }
        from = start + 3;
    }
    None
}

/// Text of the first two pages (PDF) or first few chunks, where titles and DOIs live.
fn leading_text(conn: &Connection, doc_id: i64) -> anyhow::Result<String> {
    let parts: Vec<String> = conn
        .prepare("SELECT text FROM chunks WHERE doc_id = ?1 AND (pdf_page_start IS NULL OR pdf_page_start <= 2) ORDER BY ord LIMIT 6")?
        .query_map([doc_id], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    Ok(parts.join(" "))
}

fn set_status(conn: &Connection, doc_id: i64, status: &str, candidates: Option<&str>) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE documents SET metadata_status = ?2, metadata_candidates = ?3, metadata_error = NULL, metadata_checked_at = ?4
         WHERE id = ?1 AND metadata_source IS NOT 'manual'",
        params![doc_id, status, candidates, db::now_rfc3339()],
    )?;
    Ok(())
}

/// Stores a Crossref work as the document's metadata (manual edits are never overwritten),
/// then derives printed page numbers from its page range.
pub fn apply_match(conn: &Connection, doc_id: i64, work: &Work, source: &str) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE documents SET csl_json = ?2, doi = ?3, metadata_source = ?4, metadata_status = 'matched',
           metadata_candidates = NULL, metadata_error = NULL, metadata_checked_at = ?5
         WHERE id = ?1 AND metadata_source IS NOT 'manual'",
        params![doc_id, work.to_csl().to_string(), work.doi, source, db::now_rfc3339()],
    )?;
    conn.execute("UPDATE documents SET cited_by = ?2 WHERE id = ?1 AND metadata_source IS NOT 'manual'", params![doc_id, work.cited_by])?;
    apply_page_offset(conn, doc_id)
}

/// The user's choice on the Documents tab: accept candidate `index` of a proposed match, or reject.
pub fn resolve_proposal(conn: &Connection, doc_id: i64, accept: Option<usize>) -> anyhow::Result<()> {
    let candidates: Option<String> = conn.query_row("SELECT metadata_candidates FROM documents WHERE id = ?1", [doc_id], |r| r.get(0))?;
    match accept {
        Some(i) => {
            let list: Vec<Value> = serde_json::from_str(candidates.as_deref().unwrap_or("[]"))?;
            let work: Work = serde_json::from_value(list.get(i).context("no such candidate")?["work"].clone())?;
            apply_match(conn, doc_id, &work, "crossref_query")?;
        }
        None => set_status(conn, doc_id, "unmatched", None)?,
    }
    assign_labels(conn)
}

/// Manual metadata from the Documents tab's edit form. Never overwritten by later lookups.
#[derive(Debug, Clone, Deserialize)]
pub struct ManualMetadata {
    pub title: Option<String>,
    /// "Family, Given" or a single name, one per entry.
    pub authors: Vec<String>,
    pub year: Option<i32>,
    pub doi: Option<String>,
    pub container: Option<String>,
    /// Printed page range, e.g. "1996-2005".
    pub page: Option<String>,
}

pub fn set_manual(conn: &Connection, doc_id: i64, m: &ManualMetadata) -> anyhow::Result<()> {
    let author: Vec<Value> = m
        .authors
        .iter()
        .map(|a| a.trim())
        .filter(|a| !a.is_empty())
        .map(|a| match a.split_once(',') {
            Some((family, given)) => json!({ "family": family.trim(), "given": given.trim() }),
            None => json!({ "literal": a }),
        })
        .collect();
    let mut csl = json!({ "title": m.title, "author": author, "DOI": m.doi, "container-title": m.container, "page": m.page });
    if let Some(y) = m.year {
        csl["issued"] = json!({ "date-parts": [[y]] });
    }
    conn.execute(
        "UPDATE documents SET csl_json = ?2, doi = ?3, metadata_source = 'manual', metadata_status = 'matched',
           metadata_candidates = NULL, metadata_error = NULL WHERE id = ?1",
        params![doc_id, csl.to_string(), m.doi],
    )?;
    apply_page_offset(conn, doc_id)?;
    assign_labels(conn)
}

/// When a PDF has no page labels of its own but its metadata gives a page range that fits
/// its page count (an article printed as pp. 1996–2005 with 10 PDF pages), printed numbers
/// are `pdf_page + start − 1` (§5).
pub(crate) fn apply_page_offset(conn: &Connection, doc_id: i64) -> anyhow::Result<()> {
    let (csl, page_count, has_own): (Option<String>, Option<i64>, bool) = conn.query_row(
        "SELECT csl_json, page_count, EXISTS(SELECT 1 FROM chunks WHERE doc_id = ?1 AND page_label_start IS NOT NULL
           AND (page_label_offset_applied IS NULL OR page_label_offset_applied = 0))
         FROM documents WHERE id = ?1",
        [doc_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )?;
    if has_own {
        return Ok(()); // the PDF's own labels win
    }
    let range = csl.and_then(|c| serde_json::from_str::<Value>(&c).ok()).and_then(|v| v["page"].as_str().map(str::to_string));
    let offset = range.as_deref().and_then(parse_range).and_then(|(start, end)| {
        let n = page_count?;
        // Allow one extra PDF page (a cover sheet or download stamp page).
        ((end - start + 1 == n) || (end - start + 2 == n)).then_some(start - 1)
    });
    let tx = conn.unchecked_transaction()?;
    match offset {
        Some(off) => {
            tx.execute(
                "UPDATE chunks SET page_label_start = CAST(pdf_page_start + ?2 AS TEXT), page_label_end = CAST(pdf_page_end + ?2 AS TEXT),
                   page_label_offset_applied = 1 WHERE doc_id = ?1 AND pdf_page_start IS NOT NULL",
                params![doc_id, off],
            )?;
            tx.execute("UPDATE documents SET page_label_offset = ?2 WHERE id = ?1", params![doc_id, off])?;
        }
        None => {
            tx.execute(
                "UPDATE chunks SET page_label_start = NULL, page_label_end = NULL, page_label_offset_applied = 0
                 WHERE doc_id = ?1 AND page_label_offset_applied = 1",
                [doc_id],
            )?;
            tx.execute("UPDATE documents SET page_label_offset = NULL WHERE id = ?1", [doc_id])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// "1996-2005", "1996–2005", "e123" (→ None).
fn parse_range(s: &str) -> Option<(i64, i64)> {
    let (a, b) = s.split_once(['-', '–', '—'])?;
    let (a, b): (i64, i64) = (a.trim().parse().ok()?, b.trim().parse().ok()?);
    (b >= a).then_some((a, b))
}

/// Recomputes every document's `[Author Year]` label, adding a, b, … where two documents
/// would otherwise share one (ordered by title, so the letters are stable).
pub fn assign_labels(conn: &Connection) -> anyhow::Result<()> {
    let docs: Vec<(i64, String, Option<String>)> = conn
        .prepare("SELECT id, path, csl_json FROM documents WHERE status != 'removed'")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    let mut by_base: std::collections::BTreeMap<String, Vec<(String, i64)>> = Default::default();
    for (id, path, csl) in &docs {
        let base = crate::cite::doc_label(None, csl.as_deref(), path);
        let title = csl.as_deref().and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|v| v["title"].as_str().map(str::to_lowercase)).unwrap_or_else(|| path.to_lowercase());
        by_base.entry(base).or_default().push((title, *id));
    }
    let tx = conn.unchecked_transaction()?;
    for (base, mut group) in by_base {
        group.sort();
        let suffixed = group.len() > 1 && base.chars().last().is_some_and(|c| c.is_ascii_digit());
        for (i, (_, id)) in group.iter().enumerate() {
            let label = if suffixed { format!("{base}{}", (b'a' + i.min(25) as u8) as char) } else { base.clone() };
            tx.execute("UPDATE documents SET cite_label = ?2 WHERE id = ?1", params![id, label])?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// What the Documents tab shows about a document's metadata.
pub fn summary(conn: &Connection, doc_id: i64) -> anyhow::Result<Value> {
    type Row = (Option<String>, Option<String>, Option<String>, Option<String>, Option<String>, Option<String>);
    let row: Option<Row> = conn
        .query_row(
            "SELECT csl_json, metadata_source, metadata_status, metadata_candidates, metadata_error, doi FROM documents WHERE id = ?1",
            [doc_id],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?)),
        )
        .optional()?;
    let Some((csl, source, status, candidates, error, doi)) = row else { anyhow::bail!("no document {doc_id}") };
    Ok(json!({
        "csl": csl.and_then(|c| serde_json::from_str::<Value>(&c).ok()),
        "source": source,
        "status": status,
        "candidates": candidates.and_then(|c| serde_json::from_str::<Value>(&c).ok()),
        "error": error,
        "doi": doi,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_dois_in_running_text() {
        assert_eq!(find_doi("Available at https://doi.org/10.3390/ jcm13061684 x").as_deref(), Some("10.3390/jcm13061684"));
        assert_eq!(find_doi("doi: 10.1158/1055-9965.EPI-11-0472.").as_deref(), Some("10.1158/1055-9965.epi-11-0472"));
        assert_eq!(find_doi("(see https://doi.org/10.1038/nature14539)").as_deref(), Some("10.1038/nature14539"));
        assert_eq!(find_doi("version 10.2 of the software"), None);
    }

    #[test]
    fn page_ranges() {
        assert_eq!(parse_range("1996-2005"), Some((1996, 2005)));
        assert_eq!(parse_range("345–360"), Some((345, 360)));
        assert_eq!(parse_range("e1684"), None);
    }

    #[test]
    fn titles() {
        assert!(!plausible_title("Microsoft Word - draft3.docx"));
        assert!(plausible_title("Cancer Survivors: A Booming Population"));
        assert_eq!(title_from_filename("Parry et al. - 2011 - Cancer survivors a booming population.pdf").as_deref(), Some("Cancer survivors a booming population"));
        assert!(dice("Cancer survivors: a booming population", "Cancer Survivors: A Booming Population") > 0.99);
    }
}
