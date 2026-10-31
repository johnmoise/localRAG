//! Known workspaces (`~/.config/localrag/workspaces.json`), offered in the UI's workspace
//! picker. Only paths and display names are stored; nothing about their contents.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::config::{ensure_private_dir, global_config_dir, write_private};
use crate::db;
use crate::workspace::STATE_DIR;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct KnownWorkspace {
    pub path: PathBuf,
    pub name: String,
    pub last_opened: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct KnownWorkspaceView {
    #[serde(flatten)]
    pub entry: KnownWorkspace,
    /// The folder still exists and still has `.localrag/`.
    pub available: bool,
}

fn path() -> PathBuf {
    global_config_dir().join("workspaces.json")
}

/// Most recently opened first.
pub fn list() -> Vec<KnownWorkspaceView> {
    let mut all: Vec<KnownWorkspace> =
        std::fs::read_to_string(path()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    all.sort_by(|a, b| b.last_opened.cmp(&a.last_opened));
    all.into_iter()
        .map(|entry| KnownWorkspaceView { available: entry.path.join(STATE_DIR).is_dir(), entry })
        .collect()
}

/// Adds or refreshes a workspace, marking it as just opened.
pub fn remember(root: &Path) -> anyhow::Result<()> {
    let mut all: Vec<KnownWorkspace> = list().into_iter().map(|v| v.entry).filter(|w| w.path != root).collect();
    all.push(KnownWorkspace {
        path: root.to_path_buf(),
        name: root.file_name().map_or_else(|| root.display().to_string(), |n| n.to_string_lossy().to_string()),
        last_opened: Some(db::now_rfc3339()),
    });
    save(&all)
}

/// Drops a workspace from the list. Its folder is not touched.
pub fn forget(root: &Path) -> anyhow::Result<()> {
    let all: Vec<KnownWorkspace> = list().into_iter().map(|v| v.entry).filter(|w| w.path != root).collect();
    save(&all)
}

fn save(all: &[KnownWorkspace]) -> anyhow::Result<()> {
    ensure_private_dir(&global_config_dir())?;
    write_private(&path(), serde_json::to_string_pretty(all)?.as_bytes())
}
