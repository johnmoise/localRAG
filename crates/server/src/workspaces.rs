//! Workspace picker API: list known workspaces, open, create, forget, and a native macOS
//! folder chooser (ARCHITECTURE.md §4, §15).

use std::path::PathBuf;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Deserialize;
use serde_json::json;

use localrag_core::registry;
use localrag_core::workspace::{STATE_DIR, Workspace};

use crate::AppState;
use crate::routes::{ApiError, ApiResult, blocking};

pub async fn list(State(state): State<AppState>) -> ApiResult {
    let active = state.root();
    blocking(move || {
        let known = registry::list();
        Ok(Json(json!({
            "active": active.as_ref().map(|p| json!({ "path": p, "name": p.file_name().map(|n| n.to_string_lossy().to_string()) })),
            "known": known,
            "home": std::env::var_os("HOME").map(PathBuf::from),
        })))
    })
    .await
}

#[derive(Deserialize)]
pub struct OpenBody {
    path: PathBuf,
    /// Initialize the folder as a workspace if it isn't one yet.
    #[serde(default)]
    init: bool,
}

pub async fn open(State(state): State<AppState>, Json(body): Json<OpenBody>) -> ApiResult {
    blocking(move || {
        let path = expand(&body.path);
        if !path.is_dir() {
            return Err(ApiError::new(StatusCode::NOT_FOUND, "not_found", format!("{} doesn't exist or isn't a folder", path.display())));
        }
        if !path.join(STATE_DIR).is_dir() {
            if !body.init {
                return Err(ApiError::new(
                    StatusCode::CONFLICT,
                    "not_workspace",
                    format!("{} isn't a localrag workspace yet. Initialize it (adds .localrag/ and corpus/)?", path.display()),
                ));
            }
            Workspace::init(&path)?;
        }
        let ws = state.activate(&path)?;
        Ok(Json(json!({ "active": { "path": ws.root, "name": ws.root.file_name().map(|n| n.to_string_lossy().to_string()) } })))
    })
    .await
}

#[derive(Deserialize)]
pub struct CreateBody {
    parent: PathBuf,
    name: String,
}

pub async fn create(State(state): State<AppState>, Json(body): Json<CreateBody>) -> ApiResult {
    blocking(move || {
        let root = Workspace::create(&expand(&body.parent), &body.name)
            .map_err(|e| ApiError::new(StatusCode::BAD_REQUEST, "invalid", format!("{e:#}")))?;
        let ws = state.activate(&root)?;
        Ok(Json(json!({ "active": { "path": ws.root, "name": ws.root.file_name().map(|n| n.to_string_lossy().to_string()) } })))
    })
    .await
}

#[derive(Deserialize)]
pub struct ForgetBody {
    path: PathBuf,
}

/// Removes a workspace from the known list; its folder is untouched.
pub async fn forget(State(state): State<AppState>, Json(body): Json<ForgetBody>) -> ApiResult {
    if state.root().as_ref() == Some(&body.path) {
        return Err(ApiError::new(StatusCode::CONFLICT, "active", "That workspace is open. Switch to another one first."));
    }
    blocking(move || {
        registry::forget(&body.path)?;
        Ok(Json(json!({ "forgotten": body.path })))
    })
    .await
}

/// Shows the macOS folder chooser on the user's screen and returns the chosen path.
/// The AppleScript is a fixed string: nothing from the request is interpolated into it.
pub async fn choose_folder() -> ApiResult {
    blocking(|| {
        let out = std::process::Command::new("osascript")
            .args(["-e", "activate", "-e", r#"POSIX path of (choose folder with prompt "Choose a folder for localrag")"#])
            .output()
            .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", format!("couldn't show the folder chooser: {e}")))?;
        if !out.status.success() {
            // Cancel exits with "User canceled. (-128)".
            return Ok(Json(json!({ "cancelled": true })));
        }
        let path = String::from_utf8_lossy(&out.stdout).trim().trim_end_matches('/').to_string();
        Ok(Json(json!({ "path": if path.is_empty() { "/".to_string() } else { path } })))
    })
    .await
}

/// `~/x` → `$HOME/x`, so typed paths work.
fn expand(p: &std::path::Path) -> PathBuf {
    match (p.strip_prefix("~"), std::env::var_os("HOME")) {
        (Ok(rest), Some(home)) => PathBuf::from(home).join(rest),
        _ => p.to_path_buf(),
    }
}
