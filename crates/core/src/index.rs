//! Incremental sync of `corpus/` into the index (ARCHITECTURE.md §5).
//!
//! Each file is written in its own transaction, so an interrupted sync loses at most the
//! file in progress and the next run picks up where it stopped.

use std::collections::{HashMap, HashSet};
use std::time::Instant;

use anyhow::{Context, bail};
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::json;

use crate::chunk::{self, CHUNKER_VERSION, Chunk};
use crate::db;
use crate::embed::Embedder;
use crate::parse::{self, DocKind, FileError, ParsedDoc};
use crate::scan::{self, CorpusFile};
use crate::workspace::Workspace;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pending,
    Indexed,
    NeedsOcr,
    Failed,
    Removed,
}

impl Status {
    pub fn as_str(self) -> &'static str {
        match self {
            Status::Pending => "pending",
            Status::Indexed => "indexed",
            Status::NeedsOcr => "needs_ocr",
            Status::Failed => "failed",
            Status::Removed => "removed",
        }
    }

    pub fn parse(s: &str) -> Self {
        match s {
            "indexed" => Status::Indexed,
            "needs_ocr" => Status::NeedsOcr,
            "failed" => Status::Failed,
            "removed" => Status::Removed,
            _ => Status::Pending,
        }
    }
}

/// What happened to one file during a sync.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    New,
    Changed,
    /// mtime changed but the content didn't.
    Touched,
    Removed,
    /// A removed document came back, possibly under a new path. No reprocessing.
    Restored { from: String },
    Failed(String),
}

#[derive(Debug, Clone)]
pub struct Event {
    pub path: String,
    pub change: Change,
    pub status: Status,
    pub chunks: usize,
}

#[derive(Debug, Default)]
pub struct SyncReport {
    pub new: usize,
    pub changed: usize,
    pub touched: usize,
    pub unchanged: usize,
    pub removed: usize,
    pub restored: usize,
    pub failed: usize,
    pub needs_ocr: usize,
    pub chunks_embedded: usize,
    pub chunks_reused: usize,
    /// Chunks re-embedded because the embedding model changed.
    pub reembedded: usize,
    pub too_large: Vec<(String, u64)>,
    pub millis: u128,
}

impl SyncReport {
    pub fn changed_any(&self) -> bool {
        self.new + self.changed + self.touched + self.removed + self.restored + self.failed + self.reembedded > 0
    }
}

/// Loads PDFium only if a PDF actually needs parsing.
pub struct Parsers {
    pdfium_lib: Option<std::path::PathBuf>,
}

impl Parsers {
    pub fn new(ws: &Workspace) -> Self {
        Parsers { pdfium_lib: ws.global.pdfium_lib.clone() }
    }

    fn parse(&self, file: &CorpusFile) -> anyhow::Result<Result<ParsedDoc, FileError>> {
        let bytes = std::fs::read(&file.abs_path).with_context(|| format!("reading {}", file.abs_path.display()))?;
        Ok(match file.kind {
            DocKind::Pdf => {
                let lib = parse::pdf::find_library(self.pdfium_lib.as_deref())?;
                parse::pdf::parse(parse::pdf::pdfium(&lib)?, &bytes)
            }
            DocKind::Markdown => parse::markdown::parse(&bytes),
            DocKind::Text => parse::text::parse(&bytes),
        })
    }
}

struct DocRow {
    id: i64,
    path: String,
    size: i64,
    mtime_ns: i64,
    sha256: String,
    status: Status,
    kind: String,
    parser_version: i64,
    chunker_version: i64,
    scanned_pages: usize,
}

pub fn sync(
    ws: &Workspace,
    conn: &mut Connection,
    embedder: &dyn Embedder,
    parsers: &Parsers,
    on_event: &mut dyn FnMut(&Event),
) -> anyhow::Result<SyncReport> {
    let started = Instant::now();
    let mut report = SyncReport { reembedded: ensure_embed_model(conn, embedder)?, ..SyncReport::default() };

    let scanned = scan::scan(&ws.corpus_dir(), ws.config.max_file_mb * 1024 * 1024)?;
    report.too_large = scanned.too_large;
    let rows = load_docs(conn)?;
    let mut by_path: HashMap<&str, &DocRow> = rows.iter().map(|r| (r.path.as_str(), r)).collect();
    let on_disk: HashSet<&str> = scanned.files.iter().map(|f| f.rel_path.as_str()).collect();

    // Documents that disappeared this run, plus ones removed earlier: candidates for restore.
    let mut restorable: HashMap<String, &DocRow> = HashMap::new();
    for row in &rows {
        if row.status == Status::Removed || !on_disk.contains(row.path.as_str()) {
            restorable.entry(row.sha256.clone()).or_insert(row);
        }
    }

    let mut to_process: Vec<(&CorpusFile, Option<&DocRow>, Change, String)> = Vec::new();
    let mut new_files: Vec<(&CorpusFile, String)> = Vec::new();
    for file in &scanned.files {
        match by_path.remove(file.rel_path.as_str()) {
            Some(row) => {
                let current = row.kind == file.kind.as_str()
                    && row.parser_version == file.kind.parser_version()
                    && row.chunker_version == CHUNKER_VERSION;
                let same_stat = row.size == file.size as i64 && row.mtime_ns == file.mtime_ns;
                if same_stat && current && row.status != Status::Removed {
                    report.unchanged += 1;
                    continue;
                }
                let sha = scan::sha256_file(&file.abs_path)?;
                if sha == row.sha256 && current {
                    restorable.retain(|_, r| r.id != row.id);
                    if row.status == Status::Removed {
                        restore(conn, row, file)?;
                        report.restored += 1;
                        emit(on_event, file, Change::Restored { from: row.path.clone() }, status_for(row.scanned_pages), conn, row.id)?;
                    } else {
                        update_stat(conn, row.id, file)?;
                        report.touched += 1;
                        emit(on_event, file, Change::Touched, row.status, conn, row.id)?;
                    }
                } else {
                    restorable.retain(|_, r| r.id != row.id);
                    to_process.push((file, Some(row), Change::Changed, sha));
                }
            }
            None => {
                let sha = scan::sha256_file(&file.abs_path)?;
                new_files.push((file, sha));
            }
        }
    }

    // New paths whose content matches a removed or vanished document are renames/returns.
    for (file, sha) in new_files {
        match restorable.remove(&sha) {
            Some(row) if row.kind == file.kind.as_str() => {
                restore(conn, row, file)?;
                report.restored += 1;
                emit(on_event, file, Change::Restored { from: row.path.clone() }, status_for(row.scanned_pages), conn, row.id)?;
                // Its old path is no longer "vanished".
                by_path.remove(row.path.as_str());
                if row.parser_version != file.kind.parser_version() || row.chunker_version != CHUNKER_VERSION {
                    to_process.push((file, Some(row), Change::Changed, sha));
                }
            }
            _ => to_process.push((file, None, Change::New, sha)),
        }
    }

    // Whatever is left in by_path is gone from disk.
    let mut vanished: Vec<&DocRow> = by_path.into_values().filter(|r| r.status != Status::Removed).collect();
    vanished.sort_by(|a, b| a.path.cmp(&b.path));
    for row in vanished {
        conn.execute(
            "UPDATE documents SET status = 'removed', removed_at = ?2 WHERE id = ?1",
            params![row.id, db::now_rfc3339()],
        )?;
        report.removed += 1;
        on_event(&Event { path: row.path.clone(), change: Change::Removed, status: Status::Removed, chunks: 0 });
    }

    for (file, row, change, sha) in to_process {
        let parsed = parsers.parse(file)?;
        let (status, chunks) = match parsed {
            Ok(doc) => {
                let chunks = chunk::chunk(&doc, &ws.config.chunking);
                let vectors = vectors_for(conn, embedder, &chunks, &mut report)?;
                let status = if doc.scanned_pages.is_empty() { Status::Indexed } else { Status::NeedsOcr };
                write_document(conn, file, row.map(|r| r.id), &sha, status, None, Some(&doc), &chunks, &vectors)?;
                (status, chunks.len())
            }
            Err(FileError(msg)) => {
                write_document(conn, file, row.map(|r| r.id), &sha, Status::Failed, Some(&msg), None, &[], &[])?;
                report.failed += 1;
                on_event(&Event { path: file.rel_path.clone(), change: Change::Failed(msg), status: Status::Failed, chunks: 0 });
                continue;
            }
        };
        match change {
            Change::New => report.new += 1,
            _ => report.changed += 1,
        }
        if status == Status::NeedsOcr {
            report.needs_ocr += 1;
        }
        on_event(&Event { path: file.rel_path.clone(), change, status, chunks });
    }

    db::set_meta(conn, "chunker_version", &CHUNKER_VERSION.to_string())?;
    db::set_meta(conn, "last_sync", &db::now_rfc3339())?;
    report.millis = started.elapsed().as_millis();
    Ok(report)
}

/// Removed documents only: deletes the record, chunks, vectors and OCR cache (§5).
pub fn forget(ws: &Workspace, conn: &Connection, doc_id: i64) -> anyhow::Result<String> {
    let (path, status, sha): (String, String, String) = conn
        .query_row("SELECT path, status, sha256 FROM documents WHERE id = ?1", [doc_id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .optional()?
        .with_context(|| format!("no document with id {doc_id}"))?;
    if Status::parse(&status) != Status::Removed {
        bail!("{path} is still in corpus/ ({status}); only removed documents can be forgotten. Move it out of corpus/ or add it to .localragignore");
    }
    conn.execute("DELETE FROM documents WHERE id = ?1", [doc_id])?;
    let ocr = ws.ocr_cache_path(&sha);
    if ocr.exists() {
        std::fs::remove_file(&ocr).with_context(|| format!("deleting {}", ocr.display()))?;
    }
    Ok(path)
}

/// If the configured embedding model differs from the one the index was built with, swap
/// the vector table and re-embed every chunk. Returns the number of chunks re-embedded.
fn ensure_embed_model(conn: &mut Connection, embedder: &dyn Embedder) -> anyhow::Result<usize> {
    let indexed_with = db::get_meta(conn, "embed_model")?;
    if indexed_with.as_deref() == Some(embedder.model()) {
        return Ok(0);
    }
    let dim = embedder.embed_documents(&["dimension probe".to_string()])?[0].len();
    let docs: Vec<i64> = conn.prepare("SELECT DISTINCT doc_id FROM chunks ORDER BY doc_id")?
        .query_map([], |r| r.get(0))?
        .collect::<Result<_, _>>()?;
    db::recreate_vec_table(conn, dim)?;
    let mut total = 0;
    for doc_id in docs {
        let chunks: Vec<(i64, String)> = conn.prepare("SELECT id, text FROM chunks WHERE doc_id = ?1 ORDER BY ord")?
            .query_map([doc_id], |r| Ok((r.get(0)?, r.get(1)?)))?
            .collect::<Result<_, _>>()?;
        let texts: Vec<String> = chunks.iter().map(|c| c.1.clone()).collect();
        let vectors = embedder.embed_documents(&texts)?;
        let tx = conn.transaction()?;
        for ((id, _), v) in chunks.iter().zip(&vectors) {
            tx.execute("INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)", params![id, db::vec_to_blob(v)])?;
        }
        tx.commit()?;
        total += chunks.len();
    }
    // Recorded last, so an interrupted re-embed starts over on the next run.
    db::set_meta(conn, "embed_model", embedder.model())?;
    Ok(total)
}

/// Embeddings for `chunks`, reusing stored vectors for chunk texts already in the index.
fn vectors_for(conn: &Connection, embedder: &dyn Embedder, chunks: &[Chunk], report: &mut SyncReport) -> anyhow::Result<Vec<Vec<u8>>> {
    let mut lookup = conn.prepare(
        "SELECT v.embedding FROM chunks c JOIN chunks_vec v ON v.rowid = c.id WHERE c.text_sha = ?1 LIMIT 1",
    )?;
    let mut blobs: Vec<Option<Vec<u8>>> = Vec::with_capacity(chunks.len());
    for c in chunks {
        blobs.push(lookup.query_row([&c.text_sha], |r| r.get(0)).optional()?);
    }
    let missing: Vec<usize> = (0..chunks.len()).filter(|&i| blobs[i].is_none()).collect();
    report.chunks_reused += chunks.len() - missing.len();
    report.chunks_embedded += missing.len();
    if !missing.is_empty() {
        let texts: Vec<String> = missing.iter().map(|&i| chunks[i].text.clone()).collect();
        let vectors = embedder.embed_documents(&texts)?;
        for (&i, v) in missing.iter().zip(&vectors) {
            blobs[i] = Some(db::vec_to_blob(v));
        }
    }
    Ok(blobs.into_iter().map(|b| b.expect("filled above")).collect())
}

#[allow(clippy::too_many_arguments)]
fn write_document(
    conn: &mut Connection,
    file: &CorpusFile,
    existing_id: Option<i64>,
    sha: &str,
    status: Status,
    error: Option<&str>,
    doc: Option<&ParsedDoc>,
    chunks: &[Chunk],
    vectors: &[Vec<u8>],
) -> anyhow::Result<()> {
    let tx = conn.transaction()?;
    let scanned = doc.map(|d| json!(d.scanned_pages).to_string());
    let page_count = doc.map(|d| d.pages.len() as i64);
    let (csl, doi, meta_source) = match doc {
        Some(d) if d.meta.source.is_some() => (Some(csl_json(d).to_string()), d.meta.doi.clone(), d.meta.source),
        _ => (None, doc.and_then(|d| d.meta.doi.clone()), None),
    };
    let doc_id = match existing_id {
        Some(id) => {
            tx.execute("DELETE FROM chunks WHERE doc_id = ?1", [id])?;
            tx.execute(
                "UPDATE documents SET path = ?2, size = ?3, mtime_ns = ?4, sha256 = ?5, status = ?6, kind = ?7,
                   parser_version = ?8, chunker_version = ?9, page_count = ?10, scanned_pages = ?11, error = ?12,
                   indexed_at = ?13, removed_at = NULL,
                   csl_json = CASE WHEN metadata_source = 'manual' THEN csl_json ELSE ?14 END,
                   doi = COALESCE(?15, doi),
                   metadata_source = CASE WHEN metadata_source = 'manual' THEN metadata_source ELSE ?16 END
                 WHERE id = ?1",
                params![id, file.rel_path, file.size as i64, file.mtime_ns, sha, status.as_str(), file.kind.as_str(),
                    file.kind.parser_version(), CHUNKER_VERSION, page_count, scanned, error, db::now_rfc3339(),
                    csl, doi, meta_source],
            )?;
            id
        }
        None => {
            tx.execute(
                "INSERT INTO documents (path, size, mtime_ns, sha256, status, kind, parser_version, chunker_version,
                   page_count, scanned_pages, error, indexed_at, csl_json, doi, metadata_source, metadata_status)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, 'pending')",
                params![file.rel_path, file.size as i64, file.mtime_ns, sha, status.as_str(), file.kind.as_str(),
                    file.kind.parser_version(), CHUNKER_VERSION, page_count, scanned, error, db::now_rfc3339(),
                    csl, doi, meta_source],
            )?;
            tx.last_insert_rowid()
        }
    };
    if let Some(doc) = doc {
        let mut insert_chunk = tx.prepare(
            "INSERT INTO chunks (doc_id, ord, text, text_sha, pdf_page_start, pdf_page_end, page_label_start,
               page_label_end, line_start, line_end, section, char_start, char_end, n_tokens, ocr)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15)",
        )?;
        let mut insert_vec = tx.prepare("INSERT INTO chunks_vec(rowid, embedding) VALUES (?1, ?2)")?;
        for (c, v) in chunks.iter().zip(vectors) {
            let (ps, pe) = (&doc.pages[c.page_start], &doc.pages[c.page_end]);
            insert_chunk.execute(params![
                doc_id, c.ord as i64, c.text, c.text_sha, ps.pdf_page, pe.pdf_page, ps.label, pe.label,
                c.lines.map(|l| l.0), c.lines.map(|l| l.1), (!c.section.is_empty()).then_some(&c.section),
                c.char_start as i64, c.char_end as i64, c.n_tokens as i64, c.ocr,
            ])?;
            insert_vec.execute(params![tx.last_insert_rowid(), v])?;
        }
    }
    tx.commit()?;
    Ok(())
}

fn csl_json(doc: &ParsedDoc) -> serde_json::Value {
    let m = &doc.meta;
    let mut v = json!({ "title": m.title });
    if !m.authors.is_empty() {
        v["author"] = json!(m.authors.iter().map(|a| json!({ "literal": a })).collect::<Vec<_>>());
    }
    if let Some(y) = m.year {
        v["issued"] = json!({ "date-parts": [[y]] });
    }
    if let Some(d) = &m.doi {
        v["DOI"] = json!(d);
    }
    v
}

fn restore(conn: &Connection, row: &DocRow, file: &CorpusFile) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE documents SET path = ?2, size = ?3, mtime_ns = ?4, status = ?5, removed_at = NULL WHERE id = ?1",
        params![row.id, file.rel_path, file.size as i64, file.mtime_ns, status_for(row.scanned_pages).as_str()],
    )?;
    Ok(())
}

fn update_stat(conn: &Connection, id: i64, file: &CorpusFile) -> anyhow::Result<()> {
    conn.execute(
        "UPDATE documents SET size = ?2, mtime_ns = ?3 WHERE id = ?1",
        params![id, file.size as i64, file.mtime_ns],
    )?;
    Ok(())
}

fn status_for(scanned_pages: usize) -> Status {
    if scanned_pages > 0 { Status::NeedsOcr } else { Status::Indexed }
}

fn emit(on_event: &mut dyn FnMut(&Event), file: &CorpusFile, change: Change, status: Status, conn: &Connection, id: i64) -> anyhow::Result<()> {
    let chunks: i64 = conn.query_row("SELECT count(*) FROM chunks WHERE doc_id = ?1", [id], |r| r.get(0))?;
    on_event(&Event { path: file.rel_path.clone(), change, status, chunks: chunks as usize });
    Ok(())
}

fn load_docs(conn: &Connection) -> anyhow::Result<Vec<DocRow>> {
    let mut stmt = conn.prepare(
        "SELECT id, path, size, mtime_ns, sha256, status, kind, parser_version, chunker_version, scanned_pages FROM documents",
    )?;
    let rows = stmt
        .query_map([], |r| {
            let scanned: Option<String> = r.get(9)?;
            Ok(DocRow {
                id: r.get(0)?,
                path: r.get(1)?,
                size: r.get(2)?,
                mtime_ns: r.get(3)?,
                sha256: r.get(4)?,
                status: Status::parse(&r.get::<_, String>(5)?),
                kind: r.get(6)?,
                parser_version: r.get(7)?,
                chunker_version: r.get(8)?,
                scanned_pages: scanned
                    .and_then(|s| serde_json::from_str::<Vec<u32>>(&s).ok())
                    .map_or(0, |v| v.len()),
            })
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// Counts for `localrag status`.
#[derive(Debug, Default)]
pub struct IndexStatus {
    pub by_status: Vec<(String, i64)>,
    pub chunks: i64,
    pub vectors: i64,
    pub embed_model: Option<String>,
    pub last_sync: Option<String>,
}

pub fn status(conn: &Connection) -> anyhow::Result<IndexStatus> {
    let by_status = conn
        .prepare("SELECT status, count(*) FROM documents GROUP BY status ORDER BY status")?
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?
        .collect::<Result<_, _>>()?;
    let chunks = conn.query_row("SELECT count(*) FROM chunks", [], |r| r.get(0))?;
    let has_vec: bool = conn
        .query_row("SELECT count(*) FROM sqlite_master WHERE name = 'chunks_vec'", [], |r| r.get::<_, i64>(0))
        .map(|n| n > 0)?;
    let vectors = if has_vec { conn.query_row("SELECT count(*) FROM chunks_vec", [], |r| r.get(0))? } else { 0 };
    Ok(IndexStatus {
        by_status,
        chunks,
        vectors,
        embed_model: db::get_meta(conn, "embed_model")?,
        last_sync: db::get_meta(conn, "last_sync")?,
    })
}

#[derive(Debug, Clone)]
pub struct DocSummary {
    pub id: i64,
    pub path: String,
    pub status: String,
    pub chunks: i64,
    pub error: Option<String>,
    pub scanned_pages: Option<String>,
}

pub fn list_documents(conn: &Connection) -> anyhow::Result<Vec<DocSummary>> {
    let rows = conn
        .prepare(
            "SELECT d.id, d.path, d.status, (SELECT count(*) FROM chunks c WHERE c.doc_id = d.id), d.error, d.scanned_pages
             FROM documents d ORDER BY d.path",
        )?
        .query_map([], |r| {
            Ok(DocSummary { id: r.get(0)?, path: r.get(1)?, status: r.get(2)?, chunks: r.get(3)?, error: r.get(4)?, scanned_pages: r.get(5)? })
        })?
        .collect::<Result<_, _>>()?;
    Ok(rows)
}

/// Resolves a `forget` argument: a document id or its path relative to corpus/.
pub fn find_document(conn: &Connection, key: &str) -> anyhow::Result<Option<i64>> {
    if let Ok(id) = key.parse::<i64>() {
        if conn.query_row("SELECT 1 FROM documents WHERE id = ?1", [id], |_| Ok(())).optional()?.is_some() {
            return Ok(Some(id));
        }
    }
    let path = key.trim_start_matches("corpus/");
    Ok(conn.query_row("SELECT id FROM documents WHERE path = ?1", [path], |r| r.get(0)).optional()?)
}
