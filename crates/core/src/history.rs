//! Chat history per workspace, with citation snapshots (ARCHITECTURE.md §8, §13).

use anyhow::{Context, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};

use crate::answer::{Answer, CitationStatus, Turn};
use crate::citations;
use crate::db;

/// What chat history keeps of a citation, so it stays readable after its source changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationSnapshot {
    /// Position among the answer's marker items (see `citations::placeholders`).
    #[serde(default)]
    pub ordinal: usize,
    pub marker: usize,
    /// "Parry et al. 2011, p. 1997"
    pub label: String,
    pub doc_path: String,
    pub doc_sha256: String,
    pub chunk_text_sha: String,
    pub chunk_id: i64,
    pub quote: Option<String>,
    pub quote_verified: bool,
    #[serde(default)]
    pub status: Option<CitationStatus>,
    /// Char range in the chunk text of the verified quote.
    #[serde(default)]
    pub span: Option<(usize, usize)>,
    /// When the user promoted it to Verified Paraphrase.
    #[serde(default)]
    pub paraphrase_at: Option<String>,
    /// The quote was re-selected from the passage after the answer (§10).
    #[serde(default)]
    pub quote_repaired: bool,
    /// Strict mode's verdict.
    #[serde(default)]
    pub support: Option<crate::answer::Support>,
    #[serde(default)]
    pub claim: Option<String>,
}

impl CitationSnapshot {
    pub fn status(&self) -> CitationStatus {
        self.status.unwrap_or(if self.quote_verified { CitationStatus::Verified } else { CitationStatus::Unverified })
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct SessionView {
    pub id: i64,
    pub title: String,
    pub model: Option<String>,
    pub created_at: String,
    pub messages: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct MessageView {
    pub id: i64,
    pub role: String,
    pub content: String,
    /// `content` with citation markers replaced by `⟦k⟧` placeholders.
    pub placeholder_markdown: String,
    pub citations: Vec<CitationSnapshot>,
    pub warnings: serde_json::Value,
    pub supported: Option<bool>,
    pub model: Option<String>,
    pub created_at: String,
    /// {suggestions, unsupported_sentences, invalid_markers, not_found, closest} (schema v4+).
    pub extra: serde_json::Value,
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

pub fn sessions(conn: &Connection) -> anyhow::Result<Vec<SessionView>> {
    let rows = conn
        .prepare(
            "SELECT s.id, s.title, s.model, s.created_at, (SELECT count(*) FROM messages m WHERE m.session_id = s.id)
             FROM sessions s ORDER BY s.id DESC",
        )?
        .query_map([], |r| {
            Ok(SessionView { id: r.get(0)?, title: r.get(1)?, model: r.get(2)?, created_at: r.get(3)?, messages: r.get(4)? })
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

pub fn delete_session(conn: &Connection, id: i64) -> anyhow::Result<()> {
    conn.execute("DELETE FROM sessions WHERE id = ?1", [id])?;
    Ok(())
}

pub fn messages(conn: &Connection, session: i64) -> anyhow::Result<Vec<MessageView>> {
    let rows = conn
        .prepare(
            "SELECT id, role, content, citations_json, warnings_json, supported, model, created_at, extra_json
             FROM messages WHERE session_id = ?1 ORDER BY id",
        )?
        .query_map([session], message_from_row)?
        .collect::<Result<Vec<_>, _>>()?;
    rows.into_iter().collect()
}

fn message_from_row(r: &rusqlite::Row) -> rusqlite::Result<anyhow::Result<MessageView>> {
    let content: String = r.get(2)?;
    let citations: Option<String> = r.get(3)?;
    let warnings: Option<String> = r.get(4)?;
    let id: i64 = r.get(0)?;
    let role: String = r.get(1)?;
    let supported: Option<bool> = r.get(5)?;
    let model: Option<String> = r.get(6)?;
    let created_at: String = r.get(7)?;
    let extra: Option<String> = r.get(8)?;
    Ok((|| {
        let mut snaps: Vec<CitationSnapshot> = citations.map(|c| serde_json::from_str(&c)).transpose()?.unwrap_or_default();
        backfill_ordinals(&content, &mut snaps);
        Ok(MessageView {
            id,
            placeholder_markdown: if role == "assistant" { citations::placeholders(&content) } else { content.clone() },
            content,
            role,
            citations: snaps,
            warnings: warnings.map(|w| serde_json::from_str(&w)).transpose()?.unwrap_or(serde_json::Value::Null),
            supported,
            model,
            created_at,
            extra: extra.map(|e| serde_json::from_str(&e)).transpose()?.unwrap_or(serde_json::Value::Null),
        })
    })())
}

/// Snapshots stored before ordinals existed all read as ordinal 0. Recover them by walking
/// the answer's marker items in order and pairing each with the next snapshot for that
/// source number (items naming a missing source never got a snapshot).
fn backfill_ordinals(content: &str, snaps: &mut [CitationSnapshot]) {
    if snaps.len() < 2 || snaps.iter().any(|s| s.ordinal != 0) {
        return;
    }
    let mut next = 0;
    let items = citations::parse_markers(content).into_iter().flat_map(|m| m.items);
    for (ordinal, item) in items.enumerate() {
        if next < snaps.len() && snaps[next].marker == item.source {
            snaps[next].ordinal = ordinal;
            next += 1;
        }
    }
}

pub fn message(conn: &Connection, id: i64) -> anyhow::Result<MessageView> {
    conn.query_row(
        "SELECT id, role, content, citations_json, warnings_json, supported, model, created_at, extra_json FROM messages WHERE id = ?1",
        [id],
        message_from_row,
    )
    .optional()?
    .with_context(|| format!("no message {id}"))?
}

/// Question/answer pairs of a session, oldest first.
pub fn turns(conn: &Connection, session: i64) -> anyhow::Result<Vec<Turn>> {
    let mut out = Vec::new();
    let mut pending: Option<String> = None;
    for m in messages(conn, session)? {
        match m.role.as_str() {
            "user" => pending = Some(m.content),
            "assistant" => {
                if let Some(q) = pending.take() {
                    out.push(Turn { question: q, answer: m.content });
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// Stores a question and its answer; returns the assistant message id.
pub fn record(conn: &Connection, session: i64, question: &str, answer: &Answer) -> anyhow::Result<i64> {
    let snapshots: Vec<CitationSnapshot> = answer
        .citations
        .iter()
        .map(|c| CitationSnapshot {
            ordinal: c.ordinal,
            marker: c.marker,
            label: c.chunk.citation(),
            doc_path: c.chunk.doc_path.clone(),
            doc_sha256: c.chunk.doc_sha256.clone(),
            chunk_text_sha: c.chunk.text_sha.clone(),
            chunk_id: c.chunk.id,
            quote: c.quote.clone(),
            quote_verified: c.quote_verified,
            status: Some(c.status),
            span: c.span.filter(|_| c.quote_verified),
            paraphrase_at: None,
            quote_repaired: c.quote_repaired,
            support: c.support,
            claim: Some(c.claim.clone()),
        })
        .collect();
    let now = db::now_rfc3339();
    conn.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?1, 'user', ?2, ?3)",
        params![session, question, now],
    )?;
    conn.execute(
        "INSERT INTO messages (session_id, role, content, citations_json, warnings_json, supported, model, created_at, extra_json)
         VALUES (?1, 'assistant', ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            session,
            answer.text,
            serde_json::to_string(&snapshots)?,
            serde_json::to_string(&answer.warnings)?,
            answer.supported,
            answer.model,
            now,
            serde_json::json!({
                "suggestions": answer.suggestions,
                "suggested_references": answer.suggested_references,
                "unsupported_sentences": answer.unsupported_sentences,
                "invalid_markers": answer.invalid_markers,
                "not_found": answer.not_found,
                // For "I don't know" answers: what was closest, as label + text.
                "closest": if answer.supported { Vec::new() } else {
                    answer.sources.iter().take(3).map(|c| serde_json::json!({ "label": c.citation(), "doc_label": c.doc_label, "locator": c.locator, "text": c.text, "id": c.id, "doc_id": c.doc_id, "pdf_page_start": c.pdf_page_start, "doc_path": c.doc_path })).collect()
                },
            })
            .to_string(),
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Promotes an unverified citation to Verified Paraphrase (`on`), or demotes it back, and
/// recomputes whether the answer counts as supported (§10).
pub fn set_paraphrase(conn: &Connection, message_id: i64, ordinal: usize, on: bool) -> anyhow::Result<MessageView> {
    let mut msg = message(conn, message_id)?;
    let Some(c) = msg.citations.iter_mut().find(|c| c.ordinal == ordinal) else {
        bail!("message {message_id} has no citation {ordinal}");
    };
    match (c.status(), on) {
        (CitationStatus::Unverified, true) => {
            c.status = Some(CitationStatus::VerifiedParaphrase);
            c.paraphrase_at = Some(db::now_rfc3339());
        }
        (CitationStatus::VerifiedParaphrase, false) => {
            c.status = Some(CitationStatus::Unverified);
            c.paraphrase_at = None;
        }
        (CitationStatus::Verified, _) => bail!("citation {ordinal} is already verified by its quote"),
        (s, _) => bail!("citation {ordinal} is {s:?}; nothing to change"),
    }
    let supported = msg.citations.iter().any(|c| c.status().counts_as_support() && c.support != Some(crate::answer::Support::No));
    conn.execute(
        "UPDATE messages SET citations_json = ?2, supported = ?3 WHERE id = ?1",
        params![message_id, serde_json::to_string(&msg.citations)?, supported],
    )?;
    message(conn, message_id)
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
