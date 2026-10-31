//! Chunk-reference resolution (ARCHITECTURE.md §13): does a stored citation
//! (document hash + chunk text hash + quote) still point at real, matching text?

use rusqlite::{Connection, OptionalExtension};
use serde::Serialize;

use crate::citations;
use crate::history;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Resolution {
    /// The cited document is searchable, holds the chunk, and the quote (if any) is in it.
    Resolved,
    /// The document changed, but a searchable chunk with the same text still exists.
    Relocated,
    /// The document was removed from corpus/; its chunks are kept.
    SourceRemoved,
    /// Forgotten, or the text is gone.
    Broken,
}

pub struct Resolved {
    pub resolution: Resolution,
    pub chunk_id: Option<i64>,
}

pub fn resolve(conn: &Connection, doc_sha256: &str, chunk_text_sha: &str, quote: Option<&str>) -> anyhow::Result<Resolved> {
    // Prefer the chunk inside the cited document version; fall back to any document.
    let row: Option<(i64, String, String, bool, bool)> = conn
        .query_row(
            "SELECT c.id, c.text, d.status, d.sha256 = ?1, c.ocr FROM chunks c JOIN documents d ON d.id = c.doc_id
             WHERE c.text_sha = ?2
             ORDER BY d.sha256 = ?1 DESC, d.status IN ('indexed', 'needs_ocr') DESC LIMIT 1",
            [doc_sha256, chunk_text_sha],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)),
        )
        .optional()?;
    let Some((id, text, status, same_doc, ocr)) = row else {
        return Ok(Resolved { resolution: Resolution::Broken, chunk_id: None });
    };
    if quote.is_some_and(|q| citations::verify_quote(q, &text, ocr).is_none()) {
        return Ok(Resolved { resolution: Resolution::Broken, chunk_id: Some(id) });
    }
    let resolution = match (status.as_str(), same_doc) {
        ("removed", _) => Resolution::SourceRemoved,
        ("indexed" | "needs_ocr", true) => Resolution::Resolved,
        ("indexed" | "needs_ocr", false) => Resolution::Relocated,
        _ => Resolution::Broken,
    };
    Ok(Resolved { resolution, chunk_id: Some(id) })
}

#[derive(Debug, Default, Serialize)]
pub struct CheckReport {
    pub checked: usize,
    pub resolved: usize,
    pub relocated: usize,
    pub source_removed: usize,
    /// (session, message, citation label, path)
    pub broken: Vec<(i64, i64, String, String)>,
    pub removed: Vec<(i64, i64, String, String)>,
}

impl CheckReport {
    pub fn ok(&self) -> bool {
        self.broken.is_empty()
    }
}

/// Resolves every citation in chat history. Only verified quotes are re-checked against
/// the text; unverified ones were already flagged when the answer was given.
pub fn check_history(conn: &Connection) -> anyhow::Result<CheckReport> {
    let mut report = CheckReport::default();
    for (session, message, snap) in history::all_citations(conn)? {
        report.checked += 1;
        let quote = snap.quote.as_deref().filter(|_| snap.quote_verified);
        let r = resolve(conn, &snap.doc_sha256, &snap.chunk_text_sha, quote)?;
        match r.resolution {
            Resolution::Resolved => report.resolved += 1,
            Resolution::Relocated => report.relocated += 1,
            Resolution::SourceRemoved => {
                report.source_removed += 1;
                report.removed.push((session, message, snap.label, snap.doc_path));
            }
            Resolution::Broken => report.broken.push((session, message, snap.label, snap.doc_path)),
        }
    }
    Ok(report)
}
