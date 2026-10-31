//! Chat history per workspace, with citation snapshots (ARCHITECTURE.md §8, §13).

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::answer::{Answer, Turn};
use crate::db;

/// What chat history keeps of a citation, so it stays readable after its source changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationSnapshot {
    pub marker: usize,
    /// "Parry et al. 2011, p. 1997"
    pub label: String,
    pub doc_path: String,
    pub doc_sha256: String,
    pub chunk_text_sha: String,
    pub chunk_id: i64,
    pub quote: Option<String>,
    pub quote_verified: bool,
}

pub fn create_session(conn: &Connection, title: &str, model: &str) -> anyhow::Result<i64> {
    conn.execute(
        "INSERT INTO sessions (title, model, created_at) VALUES (?1, ?2, ?3)",
        params![title.chars().take(80).collect::<String>(), model, db::now_rfc3339()],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn latest_session(conn: &Connection) -> anyhow::Result<Option<i64>> {
    Ok(conn.query_row("SELECT id FROM sessions ORDER BY id DESC LIMIT 1", [], |r| r.get(0)).optional()?)
}

pub fn session_exists(conn: &Connection, id: i64) -> anyhow::Result<bool> {
    Ok(conn.query_row("SELECT 1 FROM sessions WHERE id = ?1", [id], |_| Ok(())).optional()?.is_some())
}

/// Question/answer pairs of a session, oldest first.
pub fn turns(conn: &Connection, session: i64) -> anyhow::Result<Vec<Turn>> {
    let rows: Vec<(String, String)> = conn
        .prepare("SELECT role, content FROM messages WHERE session_id = ?1 ORDER BY id")?
        .query_map([session], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    let mut out = Vec::new();
    let mut pending: Option<String> = None;
    for (role, content) in rows {
        match role.as_str() {
            "user" => pending = Some(content),
            "assistant" => {
                if let Some(q) = pending.take() {
                    out.push(Turn { question: q, answer: content });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

pub fn record(conn: &Connection, session: i64, question: &str, answer: &Answer) -> anyhow::Result<()> {
    let snapshots: Vec<CitationSnapshot> = answer
        .citations
        .iter()
        .map(|c| CitationSnapshot {
            marker: c.marker,
            label: c.chunk.citation(),
            doc_path: c.chunk.doc_path.clone(),
            doc_sha256: c.chunk.doc_sha256.clone(),
            chunk_text_sha: c.chunk.text_sha.clone(),
            chunk_id: c.chunk.id,
            quote: c.quote.clone(),
            quote_verified: c.quote_verified,
        })
        .collect();
    let now = db::now_rfc3339();
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?1, 'user', ?2, ?3)",
        params![session, question, now],
    )?;
    conn.execute(
        "INSERT INTO messages (session_id, role, content, citations_json, warnings_json, supported, model, created_at)
         VALUES (?1, 'assistant', ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            session,
            answer.text,
            serde_json::to_string(&snapshots)?,
            serde_json::to_string(&answer.warnings)?,
            answer.supported,
            answer.model,
            now
        ],
    )?;
    Ok(())
}

/// Every citation snapshot in chat history: (session, message id, snapshot).
pub fn all_citations(conn: &Connection) -> anyhow::Result<Vec<(i64, i64, CitationSnapshot)>> {
    let rows: Vec<(i64, i64, String)> = conn
        .prepare("SELECT session_id, id, citations_json FROM messages WHERE citations_json IS NOT NULL ORDER BY id")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?
        .collect::<Result<_, _>>()?;
    let mut out = Vec::new();
    for (s, m, json) in rows {
        for snap in serde_json::from_str::<Vec<CitationSnapshot>>(&json)? {
            out.push((s, m, snap));
        }
    }
    Ok(out)
}
