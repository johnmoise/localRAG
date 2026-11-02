//! OCR inside the background job (ARCHITECTURE.md §6): scanned pages are rendered by the
//! sandboxed PDF helper, recognized with Apple Vision, cached, and the document is marked for
//! reprocessing so the following sync indexes the recognized text.

use std::path::Path;
use std::sync::atomic::AtomicBool;

use rusqlite::Connection;
use serde_json::json;

use localrag_core::index::{self, Parsers};
use localrag_core::workspace::Workspace;

use crate::AppState;

/// Set by `POST /api/ocr/cancel`; checked between pages.
pub static CANCEL: AtomicBool = AtomicBool::new(false);

pub fn run(state: &AppState, root: &Path, ws: &Workspace, conn: &Connection, parsers: &Parsers, doc_ids: &[i64]) -> anyhow::Result<()> {
    let helper = parsers.pdf_helper()?;
    for (id, path, sha, pages) in index::needing_ocr(conn)?.into_iter().filter(|d| doc_ids.contains(&d.0)) {
        let file = ws.corpus_dir().join(&path);
        let progress = |done: usize, total: usize| {
            let mut s = state.sync.lock().expect("sync state");
            if s.root.as_deref() == Some(root) {
                s.current = Some(path.clone());
                s.ocr = Some(json!({ "doc": id, "path": path, "done": done, "pages": total }));
            }
        };
        localrag_core::ocr::ocr_document(ws, helper, &file, &sha, &pages, &CANCEL, &mut { progress })?;
        index::mark_for_reprocess(conn, id)?;
    }
    Ok(())
}
