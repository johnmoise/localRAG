//! Hybrid retrieval: BM25 (FTS5) + vector kNN fused with Reciprocal Rank Fusion, capped per
//! document (ARCHITECTURE.md §9).

use std::collections::HashMap;

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

use crate::cite;
use crate::db;
use crate::embed::Embedder;

#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Candidates taken from each of BM25 and vector search.
    pub candidates: usize,
    /// Hits returned after fusion and the per-document cap.
    pub limit: usize,
    pub per_doc_cap: usize,
    pub filter: Filter,
}

impl Default for SearchOptions {
    fn default() -> Self {
        SearchOptions { candidates: 30, limit: 12, per_doc_cap: 3, filter: Filter::default() }
    }
}

/// Restricts a search to some documents and/or publication years (from their metadata).
#[derive(Debug, Clone, Default, serde::Deserialize)]
pub struct Filter {
    pub doc_ids: Option<Vec<i64>>,
    pub year_from: Option<i32>,
    pub year_to: Option<i32>,
}

impl Filter {
    pub fn is_empty(&self) -> bool {
        self.doc_ids.as_ref().is_none_or(|d| d.is_empty()) && self.year_from.is_none() && self.year_to.is_none()
    }

    /// Documents that pass, or None for "all".
    fn allowed(&self, conn: &Connection) -> anyhow::Result<Option<std::collections::HashSet<i64>>> {
        if self.is_empty() {
            return Ok(None);
        }
        let rows: Vec<(i64, Option<String>)> = conn
            .prepare("SELECT id, csl_json FROM documents")?
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        let mut ok = std::collections::HashSet::new();
        for (id, csl) in rows {
            if let Some(ids) = self.doc_ids.as_ref().filter(|d| !d.is_empty())
                && !ids.contains(&id)
            {
                continue;
            }
            if self.year_from.is_some() || self.year_to.is_some() {
                let year = csl
                    .and_then(|c| serde_json::from_str::<serde_json::Value>(&c).ok())
                    .and_then(|v| v["issued"]["date-parts"][0][0].as_i64())
                    .map(|y| y as i32);
                // Documents without a known year are excluded once a year range is set.
                let Some(y) = year else { continue };
                if self.year_from.is_some_and(|f| y < f) || self.year_to.is_some_and(|t| y > t) {
                    continue;
                }
            }
            ok.insert(id);
        }
        Ok(Some(ok))
    }
}

/// A chunk with everything needed to show and cite it.
#[derive(Debug, Clone, Serialize)]
pub struct ChunkRef {
    pub id: i64,
    pub doc_id: i64,
    pub doc_path: String,
    pub doc_sha256: String,
    pub doc_label: String,
    pub ord: i64,
    pub text: String,
    pub text_sha: String,
    /// "p. 12", "pp. 12–13", "§Methods", "ll. 4–9".
    pub locator: String,
    pub pdf_page_start: Option<i64>,
    pub section: Option<String>,
    pub ocr: bool,
    pub n_tokens: i64,
}

impl ChunkRef {
    /// "Parry et al. 2011, p. 1997"
    pub fn citation(&self) -> String {
        if self.locator.is_empty() { self.doc_label.clone() } else { format!("{}, {}", self.doc_label, self.locator) }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub chunk: ChunkRef,
    /// Fused RRF score; higher is better.
    pub score: f64,
    /// Cosine similarity to the query, when the chunk came from vector search.
    pub similarity: Option<f32>,
    pub bm25_rank: Option<usize>,
    pub vec_rank: Option<usize>,
}

const RRF_K: f64 = 60.0;

#[derive(Default)]
struct Fused {
    score: f64,
    bm25_rank: Option<usize>,
    vec_rank: Option<usize>,
    similarity: Option<f32>,
}
/// Documents whose chunks are searchable (§5): removed/failed/pending are excluded.
const SEARCHABLE: &str = "('indexed', 'needs_ocr')";

pub fn search(conn: &Connection, embedder: &dyn Embedder, query: &str, opts: &SearchOptions) -> anyhow::Result<Vec<Hit>> {
    let allowed = opts.filter.allowed(conn)?;
    if allowed.as_ref().is_some_and(|a| a.is_empty()) {
        return Ok(Vec::new());
    }
    // With a filter, draw from a much larger pool so enough candidates survive it.
    let pool = if allowed.is_some() { opts.candidates * 10 } else { opts.candidates };
    let keep = |id: &i64| allowed.as_ref().is_none_or(|a| doc_of(conn, *id).is_some_and(|d| a.contains(&d)));
    let bm25: Vec<i64> = bm25(conn, query, pool)?.into_iter().filter(|id| keep(id)).take(opts.candidates).collect();
    let query_vec = embedder.embed_query(query)?;
    let vector: Vec<(i64, f32)> = knn(conn, &query_vec, pool)?.into_iter().filter(|(id, _)| keep(id)).take(opts.candidates).collect();

    let mut fused: HashMap<i64, Fused> = HashMap::new();
    for (rank, id) in bm25.iter().enumerate() {
        let e = fused.entry(*id).or_default();
        e.score += 1.0 / (RRF_K + rank as f64 + 1.0);
        e.bm25_rank = Some(rank + 1);
    }
    for (rank, (id, sim)) in vector.iter().enumerate() {
        let e = fused.entry(*id).or_default();
        e.score += 1.0 / (RRF_K + rank as f64 + 1.0);
        e.vec_rank = Some(rank + 1);
        e.similarity = Some(*sim);
    }
    let mut ranked: Vec<(i64, Fused)> = fused.into_iter().collect();
    ranked.sort_by(|a, b| b.1.score.total_cmp(&a.1.score).then(a.0.cmp(&b.0)));

    // Soft cap: at most `per_doc_cap` chunks per document come first, so one long paper
    // can't crowd out the others; the rest backfill afterwards in score order.
    let mut per_doc: HashMap<i64, usize> = HashMap::new();
    let mut hits = Vec::new();
    let mut overflow = Vec::new();
    for (id, Fused { score, bm25_rank, vec_rank, similarity }) in ranked {
        let Some(chunk) = load_chunk(conn, id)? else { continue };
        let n = per_doc.entry(chunk.doc_id).or_default();
        *n += 1;
        let hit = Hit { chunk, score, similarity, bm25_rank, vec_rank };
        if *n <= opts.per_doc_cap { hits.push(hit) } else { overflow.push(hit) }
    }
    hits.extend(overflow);
    hits.truncate(opts.limit);
    Ok(hits)
}

fn doc_of(conn: &Connection, chunk_id: i64) -> Option<i64> {
    conn.query_row("SELECT doc_id FROM chunks WHERE id = ?1", [chunk_id], |r| r.get(0)).ok()
}

/// Neighbouring chunks (`ord ± 1`) of the same document, for context expansion.
pub fn neighbors(conn: &Connection, chunk: &ChunkRef) -> anyhow::Result<Vec<ChunkRef>> {
    let ids: Vec<i64> = conn
        .prepare("SELECT id FROM chunks WHERE doc_id = ?1 AND ord IN (?2, ?3) ORDER BY ord")?
        .query_map(params![chunk.doc_id, chunk.ord - 1, chunk.ord + 1], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    ids.into_iter().filter_map(|id| load_chunk(conn, id).transpose()).collect()
}

/// FTS5 query from free text: every word quoted (so punctuation can't break the syntax) and
/// OR-ed, letting BM25 rank by how many and how rare the matched terms are.
pub fn fts_query(text: &str) -> Option<String> {
    let terms: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() > 1 && !STOPWORDS.contains(&w.to_lowercase().as_str()))
        .map(|w| format!("\"{}\"", w.to_lowercase()))
        .collect();
    (!terms.is_empty()).then(|| terms.join(" OR "))
}

const STOPWORDS: &[&str] = &[
    "the", "and", "for", "are", "was", "were", "what", "which", "who", "whom", "how", "why", "when", "where",
    "does", "did", "do", "is", "of", "in", "on", "to", "a", "an", "by", "with", "as", "at", "be", "it",
    "that", "this", "these", "those", "from", "or", "about", "can", "has", "have", "had", "its",
];

fn bm25(conn: &Connection, query: &str, n: usize) -> anyhow::Result<Vec<i64>> {
    let Some(q) = fts_query(query) else { return Ok(Vec::new()) };
    let sql = format!(
        "SELECT f.rowid FROM chunks_fts f
         JOIN chunks c ON c.id = f.rowid JOIN documents d ON d.id = c.doc_id
         WHERE chunks_fts MATCH ?1 AND d.status IN {SEARCHABLE}
         ORDER BY bm25(chunks_fts) LIMIT ?2"
    );
    let ids = conn.prepare(&sql)?.query_map(params![q, n as i64], |r| r.get(0))?.collect::<Result<_, _>>()?;
    Ok(ids)
}

/// (chunk id, cosine similarity), best first. Over-fetches to make up for chunks of
/// non-searchable documents filtered out afterwards.
fn knn(conn: &Connection, query: &[f32], n: usize) -> anyhow::Result<Vec<(i64, f32)>> {
    let has_vec: i64 = conn.query_row("SELECT count(*) FROM sqlite_master WHERE name = 'chunks_vec'", [], |r| r.get(0))?;
    if has_vec == 0 {
        return Ok(Vec::new());
    }
    let sql = format!(
        "WITH nn AS (SELECT rowid, distance FROM chunks_vec WHERE embedding MATCH ?1 AND k = ?2)
         SELECT nn.rowid, nn.distance FROM nn
         JOIN chunks c ON c.id = nn.rowid JOIN documents d ON d.id = c.doc_id
         WHERE d.status IN {SEARCHABLE} ORDER BY nn.distance LIMIT ?3"
    );
    let rows = conn
        .prepare(&sql)?
        .query_map(params![db::vec_to_blob(query), (n * 3) as i64, n as i64], |r| {
            let d: f64 = r.get(1)?;
            Ok((r.get(0)?, (1.0 - d * d / 2.0) as f32))
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

pub fn load_chunk(conn: &Connection, id: i64) -> anyhow::Result<Option<ChunkRef>> {
    let row = conn
        .query_row(
            "SELECT c.id, c.doc_id, d.path, d.sha256, d.cite_label, d.csl_json, c.ord, c.text, c.text_sha,
                    c.page_label_start, c.page_label_end, c.pdf_page_start, c.pdf_page_end,
                    c.line_start, c.line_end, c.section, c.ocr, c.n_tokens
             FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?1",
            [id],
            |r| {
                let path: String = r.get(2)?;
                let cite_label: Option<String> = r.get(4)?;
                let csl: Option<String> = r.get(5)?;
                let (ls, le): (Option<String>, Option<String>) = (r.get(9)?, r.get(10)?);
                let section: Option<String> = r.get(15)?;
                let locator = cite::locator(
                    (ls.as_deref(), le.as_deref()),
                    (r.get(11)?, r.get(12)?),
                    (r.get(13)?, r.get(14)?),
                    section.as_deref(),
                );
                Ok(ChunkRef {
                    id: r.get(0)?,
                    doc_id: r.get(1)?,
                    doc_label: cite::doc_label(cite_label.as_deref(), csl.as_deref(), &path),
                    doc_path: path,
                    doc_sha256: r.get(3)?,
                    ord: r.get(6)?,
                    text: r.get(7)?,
                    text_sha: r.get(8)?,
                    locator,
                    pdf_page_start: r.get(11)?,
                    section,
                    ocr: r.get(16)?,
                    n_tokens: r.get(17)?,
                })
            },
        )
        .optional()?;
    Ok(row)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_query_quotes_terms_and_drops_stopwords() {
        assert_eq!(fts_query("What is the SIR of second cancers?").unwrap(), "\"sir\" OR \"second\" OR \"cancers\"");
        assert_eq!(fts_query("\"); DROP TABLE --").unwrap(), "\"drop\" OR \"table\"");
        assert!(fts_query("is the a").is_none());
    }
}
