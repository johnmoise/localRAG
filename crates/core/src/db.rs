//! The workspace database: connection setup, schema and migrations (ARCHITECTURE.md §8).

use std::path::Path;

use anyhow::Context;
use rusqlite::{Connection, OptionalExtension, ffi, params};

pub const SCHEMA_VERSION: i64 = 2;

/// Registers sqlite-vec for every connection opened afterwards in this process.
pub fn register_sqlite_vec() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| unsafe {
        #[allow(clippy::missing_transmute_annotations)]
        ffi::sqlite3_auto_extension(Some(std::mem::transmute(
            sqlite_vec::sqlite3_vec_init as *const (),
        )));
    });
}

pub fn open(path: &Path) -> anyhow::Result<Connection> {
    register_sqlite_vec();
    let conn = Connection::open(path).with_context(|| format!("opening {}", path.display()))?;
    // The busy timeout first, so the other pragmas and the migration wait out concurrent openers.
    conn.busy_timeout(std::time::Duration::from_secs(10))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    migrate(&conn)?;
    Ok(conn)
}

fn migrate(conn: &Connection) -> anyhow::Result<()> {
    if conn.pragma_query_value(None, "user_version", |r| r.get::<_, i64>(0))? == SCHEMA_VERSION {
        return Ok(());
    }
    // Several connections can open a new database at once (the background sync and UI
    // requests). BEGIN IMMEDIATE takes the write lock before the version is re-read, so
    // exactly one of them migrates and the others see the result.
    conn.execute_batch("BEGIN IMMEDIATE")?;
    let result = (|| -> anyhow::Result<()> {
        let version: i64 = conn.pragma_query_value(None, "user_version", |r| r.get(0))?;
        if version > SCHEMA_VERSION {
            anyhow::bail!("index.db has schema {version}, newer than this localrag ({SCHEMA_VERSION})");
        }
        if version < 1 {
            conn.execute_batch(SCHEMA_V1).context("creating schema v1")?;
        }
        if version < 2 {
            conn.execute_batch(SCHEMA_V2).context("migrating to schema v2")?;
        }
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(())
    })();
    match result {
        Ok(()) => conn.execute_batch("COMMIT")?,
        Err(e) => {
            let _ = conn.execute_batch("ROLLBACK");
            return Err(e);
        }
    }
    Ok(())
}

const SCHEMA_V1: &str = "
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,              -- relative to corpus/, '/'-separated
  size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, sha256 TEXT NOT NULL,
  status TEXT NOT NULL,                   -- pending|indexed|needs_ocr|failed|removed
  kind TEXT NOT NULL,                     -- pdf|markdown|text
  parser_version INTEGER NOT NULL,
  chunker_version INTEGER NOT NULL,
  page_count INTEGER, scanned_pages TEXT, -- JSON array of 1-based pdf pages needing OCR
  error TEXT, indexed_at TEXT, removed_at TEXT,
  csl_json TEXT, doi TEXT,
  metadata_source TEXT,                   -- manual|crossref_doi|crossref_query|pdf|frontmatter|heuristic
  metadata_status TEXT,                   -- pending|matched|proposed|unmatched
  cite_label TEXT, page_label_offset INTEGER
);
CREATE INDEX documents_sha ON documents(sha256);

-- char_start is an offset (in chars) into the text of page_start;
-- char_end is an offset into the text of page_end.
CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord INTEGER NOT NULL, text TEXT NOT NULL, text_sha TEXT NOT NULL,
  pdf_page_start INTEGER, pdf_page_end INTEGER, page_label_start TEXT, page_label_end TEXT,
  line_start INTEGER, line_end INTEGER,
  section TEXT, char_start INTEGER, char_end INTEGER, n_tokens INTEGER, ocr INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX chunks_doc ON chunks(doc_id, ord);
CREATE INDEX chunks_text_sha ON chunks(text_sha);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', content_rowid='id');
CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, section) VALUES (new.id, new.text, new.section);
END;
-- chunks_vec and the delete trigger are created by ensure_vec_table() once the
-- embedding dimension is known.
";

/// Chat history (§8). Citations are snapshotted so old answers stay readable after their
/// sources change or are forgotten.
const SCHEMA_V2: &str = "
CREATE TABLE sessions (id INTEGER PRIMARY KEY, title TEXT, model TEXT, created_at TEXT NOT NULL);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT NOT NULL,          -- user|assistant
  content TEXT NOT NULL,
  citations_json TEXT,         -- [{marker, label, pages, quote, quote_verified, doc_sha256, chunk_text_sha, chunk_id}]
  warnings_json TEXT,          -- context-overload warnings (§11)
  supported INTEGER, model TEXT, created_at TEXT NOT NULL
);
CREATE INDEX messages_session ON messages(session_id, id);
";

/// (Re)creates `chunks_vec` for `dim` dimensions together with the trigger that keeps it
/// and `chunks_fts` in sync on delete. Dropping it discards all vectors.
pub fn recreate_vec_table(conn: &Connection, dim: usize) -> anyhow::Result<()> {
    conn.execute_batch(&format!(
        "DROP TRIGGER IF EXISTS chunks_ad;
         DROP TABLE IF EXISTS chunks_vec;
         CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[{dim}]);
         CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
           INSERT INTO chunks_fts(chunks_fts, rowid, text, section) VALUES ('delete', old.id, old.text, old.section);
           DELETE FROM chunks_vec WHERE rowid = old.id;
         END;"
    ))?;
    set_meta(conn, "embed_dim", &dim.to_string())
}

pub fn get_meta(conn: &Connection, key: &str) -> anyhow::Result<Option<String>> {
    Ok(conn
        .query_row("SELECT value FROM meta WHERE key = ?1", [key], |r| r.get(0))
        .optional()?)
}

pub fn set_meta(conn: &Connection, key: &str, value: &str) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?1, ?2) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        params![key, value],
    )?;
    Ok(())
}

/// Little-endian f32 bytes of the unit-length vector, the blob format sqlite-vec accepts.
/// Stored vectors are always normalized so distances convert to cosine similarity.
pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut v = v.to_vec();
    crate::embed::normalize(&mut v);
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn now_rfc3339() -> String {
    humantime::format_rfc3339_seconds(std::time::SystemTime::now()).to_string()
}
