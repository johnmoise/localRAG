//! First-run Ollama setup (ARCHITECTURE.md §19.1): start Ollama at launch, report whether
//! it's installed and running, and pull the workspace's missing models with progress.

use std::path::PathBuf;
use std::time::Duration;

use axum::Json;
use axum::extract::State;
use serde::Serialize;
use serde_json::json;

use localrag_core::config::{self, WorkspaceConfig};
use localrag_core::ollama::{self, Installation, Os, PullProgress, ServeChild, Status};
use localrag_core::workspace::STATE_DIR;

use crate::AppState;
use crate::routes::{ApiResult, blocking};

/// Ollama as this server sees it.
#[derive(Debug, Default)]
pub struct OllamaState {
    /// A check (and possibly a start) is in progress.
    pub checking: bool,
    /// Something was launched and we're waiting for it to answer.
    pub starting: bool,
    /// The last check's outcome; `None` before the first finishes.
    pub status: Option<Status>,
    pub installation: Installation,
    /// An `ollama serve` this server spawned (stopped on exit only if configured).
    pub serve: Option<ServeChild>,
    pub pull: PullState,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct PullState {
    pub running: bool,
    pub model: Option<String>,
    pub progress: PullProgress,
    /// Models pulled during this run.
    pub done: Vec<String>,
    pub error: Option<String>,
}

/// Where a spawned `ollama serve` logs: the active workspace's `.localrag/logs`, else the
/// global config dir's `logs`.
fn log_dir(state: &AppState) -> PathBuf {
    match state.root() {
        Some(root) => root.join(STATE_DIR).join("logs"),
        None => config::global_config_dir().join("logs"),
    }
}

fn host() -> String {
    config::load_global().map(|g| g.ollama_host).unwrap_or_else(|_| config::GlobalConfig::default().ollama_host)
}

/// Checks Ollama in the background, starting it if it's installed, local and stopped.
/// Does nothing if a check is already running.
pub fn start_check(state: &AppState) {
    {
        let mut o = state.ollama.lock().expect("ollama state");
        if o.checking {
            return;
        }
        o.checking = true;
    }
    let state = state.clone();
    std::thread::spawn(move || {
        let host = host();
        let ensured = ollama::ensure_running(&host, &state.ollama_env, &log_dir(&state), ollama::START_TIMEOUT, &mut || {
            state.ollama.lock().expect("ollama state").starting = true;
        });
        let mut o = state.ollama.lock().expect("ollama state");
        o.checking = false;
        o.starting = false;
        o.status = Some(ensured.status);
        o.installation = ensured.installation;
        if let Some(child) = ensured.serve {
            o.serve = Some(child);
        }
    });
}

/// Blocks until the current check finishes (or `timeout` passes).
pub async fn wait_for_check(state: &AppState, timeout: Duration) {
    let deadline = tokio::time::Instant::now() + timeout;
    while state.ollama.lock().expect("ollama state").checking && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// The models this workspace uses (or the defaults with none open), with their roles.
fn wanted(state: &AppState) -> Vec<(String, &'static str)> {
    let cfg = state.workspace().map(|w| w.config).unwrap_or_else(|_| WorkspaceConfig::default());
    let mut out: Vec<(String, &'static str)> = Vec::new();
    for (name, role) in [(Some(cfg.embed_model), "embeddings"), (Some(cfg.chat_model), "answers"), (cfg.reasoning_model, "reasoning")] {
        let Some(name) = name.filter(|n| !n.trim().is_empty()) else { continue };
        if !out.iter().any(|(n, _)| *n == name) {
            out.push((name, role));
        }
    }
    out
}

/// What the setup dialog shows.
fn report(state: &AppState) -> serde_json::Value {
    let host = host();
    let (phase, status, installation, pull, started_serve) = {
        let o = state.ollama.lock().expect("ollama state");
        let phase = if o.starting { "starting" } else if o.checking || o.status.is_none() { "checking" } else { "done" };
        (phase, o.status.clone(), o.installation.clone(), o.pull.clone(), o.serve.as_ref().map(ServeChild::pid))
    };
    let running = matches!(status, Some(Status::Running { .. }));
    let wanted = wanted(state);
    let (models, models_error) = if running {
        match ollama::installed_models(&host) {
            Ok(installed) => {
                let names: Vec<&str> = wanted.iter().map(|(n, _)| n.as_str()).collect();
                let missing = ollama::missing(&names, &installed);
                let list: Vec<_> = wanted
                    .iter()
                    .map(|(n, role)| json!({ "name": n, "role": role, "installed": !missing.contains(&n.as_str()), "size": ollama::known_size(n) }))
                    .collect();
                (list, None)
            }
            Err(e) => (Vec::new(), Some(format!("{e:#}"))),
        }
    } else {
        (Vec::new(), None)
    };
    json!({
        "phase": phase,
        "host": host,
        "local": ollama::is_local_host(&host),
        "os": Os::CURRENT.name(),
        "status": status,
        "installation": installation,
        "started_serve_pid": started_serve,
        "download_url": ollama::DOWNLOAD_URL,
        "models": models,
        "models_error": models_error,
        "pull": pull,
    })
}

pub async fn get(State(state): State<AppState>) -> ApiResult {
    blocking(move || Ok(Json(report(&state)))).await
}

/// "Check again": re-probes, and starts Ollama if it has been installed meanwhile.
pub async fn check(State(state): State<AppState>) -> ApiResult {
    start_check(&state);
    blocking(move || Ok(Json(report(&state)))).await
}

/// "Download models": pulls each missing model in turn, in the background. Progress is in
/// `GET /api/ollama` → `pull`.
pub async fn pull(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let host = host();
        {
            let mut o = state.ollama.lock().expect("ollama state");
            if o.pull.running {
                drop(o);
                return Ok(Json(report(&state)));
            }
            if !matches!(o.status, Some(Status::Running { .. })) {
                o.pull = PullState { error: Some("Ollama isn't running; start it first".into()), ..PullState::default() };
                drop(o);
                return Ok(Json(report(&state)));
            }
            o.pull = PullState { running: true, ..PullState::default() };
        }
        let names: Vec<String> = wanted(&state).into_iter().map(|(n, _)| n).collect();
        let installed = ollama::installed_models(&host);
        let todo: Vec<String> = match &installed {
            Ok(i) => ollama::missing(&names.iter().map(String::as_str).collect::<Vec<_>>(), i).into_iter().map(str::to_string).collect(),
            Err(_) => Vec::new(),
        };
        if let Err(e) = installed {
            let mut o = state.ollama.lock().expect("ollama state");
            o.pull = PullState { error: Some(format!("{e:#}")), ..PullState::default() };
            drop(o);
            return Ok(Json(report(&state)));
        }
        let worker = state.clone();
        std::thread::spawn(move || {
            for model in todo {
                {
                    let mut o = worker.ollama.lock().expect("ollama state");
                    o.pull.model = Some(model.clone());
                    o.pull.progress = PullProgress { status: "starting".into(), completed: 0, total: ollama::known_size(&model).unwrap_or(0) };
                }
                let r = ollama::pull(&host, &model, &mut |p| {
                    worker.ollama.lock().expect("ollama state").pull.progress = p.clone();
                });
                let mut o = worker.ollama.lock().expect("ollama state");
                match r {
                    Ok(()) => o.pull.done.push(model),
                    Err(e) => {
                        o.pull.error = Some(format!("{e:#}"));
                        break;
                    }
                }
            }
            let mut o = worker.ollama.lock().expect("ollama state");
            o.pull.running = false;
            o.pull.model = None;
        });
        Ok(Json(report(&state)))
    })
    .await
}

/// On shutdown: stops an `ollama serve` we started, if `stop_ollama_on_exit` is set.
pub fn on_exit(state: &AppState) {
    let stop = config::load_global().map(|g| g.stop_ollama_on_exit).unwrap_or(false);
    let child = state.ollama.lock().expect("ollama state").serve.take();
    // Otherwise dropping the handle leaves it running for next time.
    if let Some(child) = child
        && stop
    {
        child.stop();
    }
}
