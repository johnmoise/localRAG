//! The localhost web server (ARCHITECTURE.md §14): token-guarded JSON API, SSE answers, the
//! embedded web UI, and a background index sync.
//!
//! Core work (SQLite, PDFium, Ollama) is blocking, so every handler runs it on Tokio's
//! blocking pool and opens its own database connection; WAL mode lets them run alongside the
//! background sync.

mod auth;
mod routes;

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use axum::Router;
use serde::Serialize;

use localrag_core::db;
use localrag_core::embed::OllamaEmbedder;
use localrag_core::index::{self, Change, Parsers, SyncReport};
use localrag_core::workspace::Workspace;

pub use auth::TOKEN_COOKIE;

#[derive(Clone)]
pub struct AppState {
    pub root: PathBuf,
    pub token: String,
    pub port: u16,
    pub sync: Arc<Mutex<SyncState>>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct SyncState {
    pub running: bool,
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
    pub fn new(root: &Path, port: u16, token: String) -> Self {
        AppState { root: root.to_path_buf(), token, port, sync: Arc::default() }
    }

    /// The workspace, re-read so config edits take effect without a restart.
    pub fn workspace(&self) -> anyhow::Result<Workspace> {
        Workspace::open(&self.root)
    }

    /// Starts an index sync in the background unless one is already running.
    pub fn start_sync(&self) -> bool {
        {
            let mut s = self.sync.lock().expect("sync state");
            if s.running {
                return false;
            }
            *s = SyncState { running: true, runs: s.runs, last: s.last.clone(), ..SyncState::default() };
        }
        let state = self.clone();
        std::thread::spawn(move || {
            let result = (|| -> anyhow::Result<SyncReport> {
                let ws = state.workspace()?;
                let mut conn = db::open(&ws.db_path())?;
                let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
                index::sync(&ws, &mut conn, &embedder, &Parsers::new(&ws), &mut |e| {
                    let mut s = state.sync.lock().expect("sync state");
                    if !matches!(e.change, Change::Touched) {
                        s.processed += 1;
                    }
                    s.current = Some(e.path.clone());
                })
            })();
            let mut s = state.sync.lock().expect("sync state");
            s.running = false;
            s.current = None;
            s.runs += 1;
            match result {
                Ok(r) => s.last = Some(SyncSummary::from(&r)),
                Err(e) => s.error = Some(format!("{e:#}")),
            }
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

/// Serves the workspace at `root` until the process exits. Calls `on_ready` with the
/// launch URL (which carries the access token) once listening.
pub async fn serve(root: &Path, port: u16, on_ready: impl FnOnce(&str)) -> anyhow::Result<()> {
    let listener = bind(port).await?;
    let port = listener.local_addr()?.port();
    let state = AppState::new(root, port, new_token()?);
    state.start_sync();
    on_ready(&format!("http://127.0.0.1:{port}/?token={}", state.token));
    axum::serve(listener, router(state)).await?;
    Ok(())
}
