//! The fact registry (§13): `brain_facts` is the source of truth for fact IDs, per model.
//! IDs are assigned monotonically and never reused; deleted facts stay as tombstones.
//! Also the fact / rejected-fact embeddings used by the admission rules.

use rusqlite::{Connection, OptionalExtension, params};
use serde::Serialize;

use super::file::{FactCitation, FactLine, id_number, render_fact};
use crate::db;
use crate::embed::Embedder;
use crate::scan::sha256_str;

#[derive(Debug, Clone, Serialize)]
pub struct Registered {
    pub id: String,
    pub line: String,
    pub line_sha: String,
    pub claim: String,
    pub status: String,
    pub added_at: String,
    pub reason: Option<String>,
    pub citations: Vec<FactCitation>,
}

pub fn next_id(conn: &Connection, model: &str) -> anyhow::Result<String> {
    let ids: Vec<String> = conn.prepare("SELECT fact_id FROM brain_facts WHERE model = ?1")?.query_map([model], |r| r.get(0))?.collect::<Result<_, _>>()?;
    let n = ids.iter().filter_map(|i| id_number(i)).max().unwrap_or(0) + 1;
    Ok(format!("F-{n:04}"))
}

/// Registers a new fact and returns its line (not yet in the file).
pub fn register(conn: &Connection, model: &str, claim: &str, citations: &[FactCitation], reason: Option<&str>, run_id: Option<i64>) -> anyhow::Result<FactLine> {
    let tx = conn.unchecked_transaction()?;
    let id = next_id(&tx, model)?;
    let fact = FactLine { id: id.clone(), claim: claim.trim().to_string(), citations: citations.to_vec() };
    let line = render_fact(&fact);
    tx.execute(
        "INSERT INTO brain_facts (model, fact_id, line, line_sha, claim, status, added_at, run_id, reason) VALUES (?1, ?2, ?3, ?4, ?5, 'active', ?6, ?7, ?8)",
        params![model, id, line, sha256_str(&line), fact.claim, db::now_rfc3339(), run_id, reason],
    )?;
    write_citations(&tx, model, &id, citations)?;
    tx.commit()?;
    Ok(fact)
}

/// Replaces a registered fact's line (merge, edit, accepted hand edit).
pub fn rewrite(conn: &Connection, model: &str, fact: &FactLine) -> anyhow::Result<String> {
    let line = render_fact(fact);
    let tx = conn.unchecked_transaction()?;
    tx.execute(
        "UPDATE brain_facts SET line = ?3, line_sha = ?4, claim = ?5 WHERE model = ?1 AND fact_id = ?2",
        params![model, fact.id, line, sha256_str(&line), fact.claim],
    )?;
    tx.execute("DELETE FROM fact_citations WHERE model = ?1 AND fact_id = ?2", params![model, fact.id])?;
    write_citations(&tx, model, &fact.id, &fact.citations)?;
    tx.commit()?;
    Ok(line)
}

fn write_citations(conn: &Connection, model: &str, id: &str, citations: &[FactCitation]) -> anyhow::Result<()> {
    for (i, c) in citations.iter().enumerate() {
        conn.execute(
            "INSERT INTO fact_citations (model, fact_id, ord, label, doc_sha256, chunk_text_sha, quote) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![model, id, i as i64, c.label, c.doc_sha256, c.chunk_text_sha, c.quote],
        )?;
    }
    Ok(())
}

pub fn get(conn: &Connection, model: &str, id: &str) -> anyhow::Result<Option<Registered>> {
    let row = conn
        .query_row(
            "SELECT fact_id, line, line_sha, claim, status, added_at, reason FROM brain_facts WHERE model = ?1 AND fact_id = ?2",
            params![model, id],
            |r| Ok(Registered { id: r.get(0)?, line: r.get(1)?, line_sha: r.get(2)?, claim: r.get(3)?, status: r.get(4)?, added_at: r.get(5)?, reason: r.get(6)?, citations: Vec::new() }),
        )
        .optional()?;
    let Some(mut f) = row else { return Ok(None) };
    f.citations = citations(conn, model, id)?;
    Ok(Some(f))
}

pub fn citations(conn: &Connection, model: &str, id: &str) -> anyhow::Result<Vec<FactCitation>> {
    Ok(conn
        .prepare("SELECT label, quote, doc_sha256, chunk_text_sha FROM fact_citations WHERE model = ?1 AND fact_id = ?2 ORDER BY ord")?
        .query_map(params![model, id], |r| Ok(FactCitation { label: r.get(0)?, quote: r.get(1)?, doc_sha256: r.get(2)?, chunk_text_sha: r.get(3)? }))?
        .collect::<Result<_, _>>()?)
}

pub fn all(conn: &Connection, model: &str) -> anyhow::Result<Vec<Registered>> {
    let ids: Vec<String> = conn.prepare("SELECT fact_id FROM brain_facts WHERE model = ?1 ORDER BY fact_id")?.query_map([model], |r| r.get(0))?.collect::<Result<_, _>>()?;
    ids.iter().filter_map(|id| get(conn, model, id).transpose()).collect()
}

pub fn set_status(conn: &Connection, model: &str, id: &str, status: &str) -> anyhow::Result<()> {
    conn.execute("UPDATE brain_facts SET status = ?3 WHERE model = ?1 AND fact_id = ?2 AND status != 'deleted'", params![model, id, status])?;
    Ok(())
}

/// Tombstones a fact and records it as rejected, so it isn't proposed again.
pub fn delete(conn: &Connection, embedder: &dyn Embedder, model: &str, id: &str) -> anyhow::Result<()> {
    let Some(f) = get(conn, model, id)? else { anyhow::bail!("no fact {id}") };
    conn.execute("UPDATE brain_facts SET status = 'deleted' WHERE model = ?1 AND fact_id = ?2", params![model, id])?;
    conn.execute("DELETE FROM fact_links WHERE model = ?1 AND (a = ?2 OR b = ?2)", params![model, id])?;
    forget_embedding(conn, model, "fact", id)?;
    reject(conn, embedder, model, &f.claim, &f.citations, "deleted")?;
    Ok(())
}

/// Records a rejected claim (and embeds it for the "not previously rejected" rule).
pub fn reject(conn: &Connection, embedder: &dyn Embedder, model: &str, claim: &str, citations: &[FactCitation], origin: &str) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO rejected_facts (model, claim, citations_json, origin, at) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![model, claim, serde_json::to_string(citations)?, origin, db::now_rfc3339()],
    )?;
    let id = conn.last_insert_rowid().to_string();
    embed(conn, embedder, model, "rejected", &id, claim)
}

pub fn link_conflict(conn: &Connection, model: &str, a: &str, b: &str) -> anyhow::Result<()> {
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    conn.execute("INSERT OR IGNORE INTO fact_links (model, a, b, kind) VALUES (?1, ?2, ?3, 'conflicts')", params![model, a, b])?;
    Ok(())
}

/// Facts linked to `id` as conflicting.
pub fn conflicts(conn: &Connection, model: &str, id: &str) -> anyhow::Result<Vec<String>> {
    Ok(conn
        .prepare("SELECT CASE WHEN a = ?2 THEN b ELSE a END FROM fact_links WHERE model = ?1 AND kind = 'conflicts' AND (a = ?2 OR b = ?2)")?
        .query_map(params![model, id], |r| r.get(0))?
        .collect::<Result<_, _>>()?)
}

/// Stores the embedding of `text` for (model, kind, ref), replacing an older one.
pub fn embed(conn: &Connection, embedder: &dyn Embedder, model: &str, kind: &str, r: &str, text: &str) -> anyhow::Result<()> {
    let sha = sha256_str(text);
    let existing: Option<(i64, String)> = conn
        .query_row("SELECT id, text_sha FROM brain_embeddings WHERE model = ?1 AND kind = ?2 AND ref = ?3", params![model, kind, r], |row| Ok((row.get(0)?, row.get(1)?)))
        .optional()?;
    if existing.as_ref().is_some_and(|(_, s)| *s == sha) && db::has_table(conn, "brain_vec")? {
        return Ok(());
    }
    let v = embedder.embed_documents(&[text.to_string()])?.pop().unwrap_or_default();
    db::ensure_brain_vec(conn, v.len())?;
    let id = match existing {
        Some((id, _)) => {
            conn.execute("UPDATE brain_embeddings SET text_sha = ?2 WHERE id = ?1", params![id, sha])?;
            conn.execute("DELETE FROM brain_vec WHERE rowid = ?1", [id])?;
            id
        }
        None => {
            conn.execute("INSERT INTO brain_embeddings (model, kind, ref, text_sha) VALUES (?1, ?2, ?3, ?4)", params![model, kind, r, sha])?;
            conn.last_insert_rowid()
        }
    };
    conn.execute("INSERT INTO brain_vec(rowid, embedding) VALUES (?1, ?2)", params![id, db::vec_to_blob(&v)])?;
    Ok(())
}

pub fn forget_embedding(conn: &Connection, model: &str, kind: &str, r: &str) -> anyhow::Result<()> {
    let id: Option<i64> = conn.query_row("SELECT id FROM brain_embeddings WHERE model = ?1 AND kind = ?2 AND ref = ?3", params![model, kind, r], |row| row.get(0)).optional()?;
    if let Some(id) = id {
        conn.execute("DELETE FROM brain_embeddings WHERE id = ?1", [id])?;
        if db::has_table(conn, "brain_vec")? {
            conn.execute("DELETE FROM brain_vec WHERE rowid = ?1", [id])?;
        }
    }
    Ok(())
}

/// The `k` nearest (ref, cosine similarity) of `kind` for this model.
pub fn nearest(conn: &Connection, embedder: &dyn Embedder, model: &str, kind: &str, text: &str, k: usize) -> anyhow::Result<Vec<(String, f32)>> {
    if !db::has_table(conn, "brain_vec")? {
        return Ok(Vec::new());
    }
    let v = embedder.embed_documents(&[text.to_string()])?.pop().unwrap_or_default();
    nearest_vec(conn, &v, model, kind, k)
}

pub fn nearest_vec(conn: &Connection, v: &[f32], model: &str, kind: &str, k: usize) -> anyhow::Result<Vec<(String, f32)>> {
    if !db::has_table(conn, "brain_vec")? {
        return Ok(Vec::new());
    }
    let total: i64 = conn.query_row("SELECT count(*) FROM brain_embeddings", [], |r| r.get(0))?;
    if total == 0 {
        return Ok(Vec::new());
    }
    let mut out: Vec<(String, f32)> = conn
        .prepare(
            "WITH nn AS (SELECT rowid, distance FROM brain_vec WHERE embedding MATCH ?1 AND k = ?2)
             SELECT e.ref, nn.distance FROM nn JOIN brain_embeddings e ON e.id = nn.rowid
             WHERE e.model = ?3 AND e.kind = ?4 ORDER BY nn.distance",
        )?
        .query_map(params![db::vec_to_blob(v), total.min(4096), model, kind], |r| Ok((r.get::<_, String>(0)?, r.get::<_, f64>(1)?)))?
        .map(|r| r.map(|(id, d)| (id, (1.0 - d * d / 2.0) as f32)))
        .collect::<Result<_, _>>()?;
    out.truncate(k);
    Ok(out)
}
