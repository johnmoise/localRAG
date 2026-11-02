use anyhow::{Context, ensure};
use rusqlite::Connection;

use crate::db::register_sqlite_vec;

use super::Outcome;

/// FTS5 and vec0 work, and the chunk triggers leave no orphans after a cascaded delete (§8).
pub(super) fn check() -> anyhow::Result<Outcome> {
    register_sqlite_vec();
    let db = Connection::open_in_memory()?;
    let (sqlite_version, vec_version): (String, String) =
        db.query_row("SELECT sqlite_version(), vec_version()", [], |r| Ok((r.get(0)?, r.get(1)?)))
            .context("sqlite-vec did not load")?;

    db.execute_batch(
        "PRAGMA foreign_keys = ON;
         CREATE TABLE documents (id INTEGER PRIMARY KEY, path TEXT);
         CREATE TABLE chunks (
           id INTEGER PRIMARY KEY,
           doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
           text TEXT, section TEXT);
         CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', content_rowid='id');
         CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[4]);
         CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
           INSERT INTO chunks_fts(rowid, text, section) VALUES (new.id, new.text, new.section);
         END;
         CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
           INSERT INTO chunks_fts(chunks_fts, rowid, text, section) VALUES ('delete', old.id, old.text, old.section);
           DELETE FROM chunks_vec WHERE rowid = old.id;
         END;
         INSERT INTO documents VALUES (1, 'a.pdf'), (2, 'b.pdf');
         INSERT INTO chunks VALUES
           (10, 1, 'self-attention is quadratic', 'Intro'),
           (11, 1, 'linear attention is cheaper', 'Methods'),
           (20, 2, 'unrelated text about birds', 'Intro');
         INSERT INTO chunks_vec(rowid, embedding) VALUES
           (10, '[1, 0, 0, 0]'), (11, '[0.9, 0.1, 0, 0]'), (20, '[0, 0, 1, 0]');",
    )
    .context("FTS5 / vec0 schema")?;

    let fts_hit: i64 = db.query_row(
        "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH 'quadratic'", [], |r| r.get(0))?;
    ensure!(fts_hit == 10, "FTS5 returned rowid {fts_hit}, expected 10");

    let knn: Vec<i64> = db
        .prepare("SELECT rowid FROM chunks_vec WHERE embedding MATCH '[1, 0, 0, 0]' AND k = 2 ORDER BY distance")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    ensure!(knn == [10, 11], "vector kNN returned {knn:?}, expected [10, 11]");

    db.execute("DELETE FROM documents WHERE id = 1", [])?;
    let orphans_fts: i64 = db.query_row(
        "SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'attention'", [], |r| r.get(0))?;
    let vec_rows: i64 = db.query_row("SELECT count(*) FROM chunks_vec", [], |r| r.get(0))?;
    ensure!(orphans_fts == 0, "{orphans_fts} FTS rows left after cascaded delete");
    ensure!(vec_rows == 1, "{} vector rows left after cascaded delete", vec_rows - 1);

    Ok(Outcome::Pass(format!(
        "SQLite {sqlite_version}, sqlite-vec {vec_version}; FTS5 match, kNN and cascade triggers OK"
    )))
}

#[cfg(test)]
mod tests {
    #[test]
    fn fts_vec_and_cascade_triggers() {
        super::check().unwrap();
    }
}
