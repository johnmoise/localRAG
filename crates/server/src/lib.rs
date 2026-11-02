//! The localhost web server (ARCHITECTURE.md §14): token-guarded JSON API, SSE answers, the
//! embedded web UI, and a background index sync.
//!
//! Core work (SQLite, PDFium, Ollama) is blocking, so every handler runs it on Tokio's
//! blocking pool and opens its own database connection; WAL mode lets them run alongside the
//! background sync.

mod auth;
mod brain;
mod ocr_jobs;
mod routes;
mod settings;
mod upload;
mod workspaces;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, RwLock};

use anyhow::Context;
use axum::Router;
use serde::Serialize;

use localrag_core::db;
use localrag_core::embed::OllamaEmbedder;
use localrag_core::index::{self, Change, Parsers, SyncReport};
use localrag_core::workspace::Workspace;

pub use auth::TOKEN_COOKIE;

/// Server state. Exactly one workspace is active at a time (or none, until the user picks
/// one); switching replaces it for every route.
#[derive(Clone)]
pub struct AppState {
    pub token: String,
    pub port: u16,
    active: Arc<RwLock<Option<PathBuf>>>,
    pub sync: Arc<Mutex<SyncState>>,
}

/// No workspace is open yet: the UI shows the workspace picker.
#[derive(Debug)]
pub struct NoWorkspace;

impl std::fmt::Display for NoWorkspace {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("no workspace is open; choose or create one")
    }
}

impl std::error::Error for NoWorkspace {}

#[derive(Debug, Default, Clone, Serialize)]
pub struct SyncState {
    /// The workspace this state describes.
    pub root: Option<PathBuf>,
    pub running: bool,
    /// What the background job is doing: "indexing", "metadata" or "ocr".
    pub phase: Option<String>,
    /// Metadata results of the last run (matched / proposed / unmatched / errors).
    pub metadata: Option<serde_json::Value>,
    /// OCR progress: {doc, path, page, pages}.
    pub ocr: Option<serde_json::Value>,
    /// Another sync was asked for while one ran (e.g. an upload); it starts right after.
    pub rerun: bool,
    /// Completed syncs since the server started; lets the UI tell when "its" sync is done.
    pub runs: u64,
    pub processed: usize,
    pub current: Option<String>,
    pub last: Option<SyncSummary>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SyncSummary {
    pub new: usize,
    pub changed: usize,
    pub removed: usize,
    pub restored: usize,
    pub failed: usize,
    pub unchanged: usize,
    pub needs_ocr: usize,
    pub millis: u128,
    pub finished_at: String,
}

impl From<&SyncReport> for SyncSummary {
    fn from(r: &SyncReport) -> Self {
        SyncSummary {
            new: r.new,
            changed: r.changed,
            removed: r.removed,
            restored: r.restored,
            failed: r.failed,
            unchanged: r.unchanged,
            needs_ocr: r.needs_ocr,
            millis: r.millis,
            finished_at: db::now_rfc3339(),
        }
    }
}

impl AppState {
    pub fn new(root: Option<&Path>, port: u16, token: String) -> Self {
        AppState { token, port, active: Arc::new(RwLock::new(root.map(Path::to_path_buf))), sync: Arc::default() }
    }

    pub fn root(&self) -> Option<PathBuf> {
        self.active.read().expect("active workspace").clone()
    }

    /// The active workspace, re-read so config edits take effect without a restart.
    pub fn workspace(&self) -> anyhow::Result<Workspace> {
        let root = self.root().ok_or(NoWorkspace)?;
        Workspace::open(&root)
    }

    /// Makes `root` the one active workspace and remembers it. Indexing is manual (an upload
    /// or Reindex), so opening a workspace doesn't sync it.
    pub fn activate(&self, root: &Path) -> anyhow::Result<Workspace> {
        let ws = Workspace::open(root)?;
        localrag_core::registry::remember(&ws.root)?;
        // One app instance per machine: a run still marked running was cut off by a restart.
        // (A run in the workspace being left keeps going and records its own result.)
        if self.root().as_deref() != Some(ws.root.as_path()) {
            localrag_core::agent::fail_interrupted(&db::open(&ws.db_path())?)?;
        }
        *self.active.write().expect("active workspace") = Some(ws.root.clone());
        {
            let mut s = self.sync.lock().expect("sync state");
            // A sync still running for the previous workspace finishes on its own; its
            // result is dropped because it no longer matches `root`.
            *s = SyncState { root: Some(ws.root.clone()), ..SyncState::default() };
        }
        Ok(ws)
    }

    /// Starts an index sync of the active workspace in the background. If one is already
    /// running, another is queued to start when it ends (so a just-uploaded file is picked up).
    pub fn start_sync(&self) -> bool {
        self.start_job(Vec::new())
    }

    /// OCRs these documents' scanned pages, then syncs (which picks up the OCR text). Refused
    /// while another job runs.
    pub fn start_ocr(&self, doc_ids: Vec<i64>) -> bool {
        if self.sync.lock().expect("sync state").running {
            return false;
        }
        self.start_job(doc_ids)
    }

    /// The one background worker: optional OCR → indexing → Crossref metadata → (auto OCR).
    fn start_job(&self, mut ocr_first: Vec<i64>) -> bool {
        let Some(root) = self.root() else { return false };
        {
            let mut s = self.sync.lock().expect("sync state");
            if s.running && s.root.as_ref() == Some(&root) {
                s.rerun = true;
                return false;
            }
            *s = SyncState { root: Some(root.clone()), running: true, runs: s.runs, last: s.last.clone(), ..SyncState::default() };
        }
        crate::ocr_jobs::CANCEL.store(false, std::sync::atomic::Ordering::SeqCst);
        let state = self.clone();
        std::thread::spawn(move || loop {
            let set_phase = |phase: &str| {
                let mut s = state.sync.lock().expect("sync state");
                if s.root.as_ref() == Some(&root) {
                    s.phase = Some(phase.to_string());
                }
            };
            let result = (|| -> anyhow::Result<SyncReport> {
                let ws = Workspace::open(&root)?;
                let mut conn = db::open(&ws.db_path())?;
                let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
                let parsers = Parsers::new(&ws);
                if !ocr_first.is_empty() {
                    set_phase("ocr");
                    crate::ocr_jobs::run(&state, &root, &ws, &conn, &parsers, &std::mem::take(&mut ocr_first))?;
                }
                set_phase("indexing");
                let mut report = index::sync(&ws, &mut conn, &embedder, &parsers, &mut |e| {
                    let mut s = state.sync.lock().expect("sync state");
                    if s.root.as_ref() == Some(&root) {
                        if !matches!(e.change, Change::Touched) {
                            s.processed += 1;
                        }
                        s.current = Some(e.path.clone());
                    }
                })?;
                // Then citation metadata for new documents (network; never fails the sync).
                set_phase("metadata");
                let client = localrag_core::metadata::crossref::configured(&ws.global);
                let meta = localrag_core::metadata::process_pending(&conn, client, &mut |e| {
                    let mut s = state.sync.lock().expect("sync state");
                    if s.root.as_ref() == Some(&root) {
                        s.current = Some(e.path.clone());
                    }
                });
                let meta = match meta {
                    Ok(r) => serde_json::to_value(&r).unwrap_or_default(),
                    Err(e) => serde_json::json!({ "errors": [format!("{e:#}")] }),
                };
                state.sync.lock().expect("sync state").metadata = Some(meta);
                // References from bibliographies: Crossref enrichment, then corpus links.
                let _ = localrag_core::references::backfill_cited_by(&conn, client);
                set_phase("references");
                let refs = localrag_core::references::enrich_pending(&conn, client, &mut |done, total| {
                    let mut s = state.sync.lock().expect("sync state");
                    if s.root.as_ref() == Some(&root) {
                        s.current = Some(format!("{done} of {total} references"));
                    }
                });
                if refs.is_err() {
                    localrag_core::references::link(&conn)?;
                }
                // Abstracts for references matched before they were kept, then one embedded
                // chunk per reference (for suggesting additions; never an answer source).
                let _ = localrag_core::references::backfill_details(&conn, client, &mut |done, total| {
                    let mut s = state.sync.lock().expect("sync state");
                    if s.root.as_ref() == Some(&root) {
                        s.current = Some(format!("abstracts: {done} of {total} references"));
                    }
                });
                {
                    let mut s = state.sync.lock().expect("sync state");
                    if s.root.as_ref() == Some(&root) {
                        s.current = Some("embedding references".into());
                    }
                }
                localrag_core::references::embed_pending(&conn, &embedder)?;
                // OCR runs on its own only when the workspace asks for it (ocr = "auto").
                if ws.config.ocr == localrag_core::config::OcrMode::Auto {
                    let waiting: Vec<i64> = index::needing_ocr(&conn)?.into_iter().map(|d| d.0).collect();
                    if !waiting.is_empty() {
                        set_phase("ocr");
                        crate::ocr_jobs::run(&state, &root, &ws, &conn, &parsers, &waiting)?;
                        set_phase("indexing");
                        report = index::sync(&ws, &mut conn, &embedder, &parsers, &mut |_| {})?;
                    }
                }
                Ok(report)
            })();
            let mut s = state.sync.lock().expect("sync state");
            if s.root.as_ref() != Some(&root) {
                return; // the user switched workspaces meanwhile
            }
            s.current = None;
            s.phase = None;
            s.ocr = None;
            s.runs += 1;
            match result {
                Ok(r) => {
                    s.last = Some(SyncSummary::from(&r));
                    s.error = None;
                }
                Err(e) => s.error = Some(format!("{e:#}")),
            }
            if std::mem::take(&mut s.rerun) {
                s.processed = 0;
                continue;
            }
            s.running = false;
            return;
        });
        true
    }
}

pub fn router(state: AppState) -> Router {
    routes::router(state.clone()).layer(axum::middleware::from_fn_with_state(state, auth::guard))
}

/// A random 32-byte token, hex-encoded, from the OS RNG.
pub fn new_token() -> anyhow::Result<String> {
    use std::io::Read;
    let mut bytes = [0u8; 32];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}

/// Binds `127.0.0.1:port` (or a free port if it's taken or 0).
pub async fn bind(port: u16) -> anyhow::Result<tokio::net::TcpListener> {
    let addr = SocketAddr::from(([127, 0, 0, 1], port));
    match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => Ok(l),
        Err(_) if port != 0 => Ok(tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0))).await?),
        Err(e) => Err(e).context("binding 127.0.0.1"),
    }
}

/// Serves until the process exits, with `root` active (or none, so the UI starts on the
/// workspace picker). Calls `on_ready` with the launch URL, which carries the access token.
pub async fn serve(root: Option<&Path>, port: u16, on_ready: impl FnOnce(&str)) -> anyhow::Result<()> {
    let listener = bind(port).await?;
    let port = listener.local_addr()?.port();
    let state = AppState::new(None, port, new_token()?);
    if let Some(root) = root {
        state.activate(root)?;
    }
    on_ready(&format!("http://127.0.0.1:{port}/?token={}", state.token));
    axum::serve(listener, router(state)).await?;
    Ok(())
}
