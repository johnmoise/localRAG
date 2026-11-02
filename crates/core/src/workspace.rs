//! Finding and creating workspaces (ARCHITECTURE.md §4).

use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

use crate::config::{self, GlobalConfig, WorkspaceConfig};

pub const STATE_DIR: &str = ".localrag";
pub const CORPUS_DIR: &str = "corpus";
pub const IGNORE_FILE: &str = ".localragignore";

const CONFIG_HEADER: &str = "# localrag workspace settings. See ARCHITECTURE.md §4.\n\n";

#[derive(Debug, Clone)]
pub struct Workspace {
    pub root: PathBuf,
    pub config: WorkspaceConfig,
    pub global: GlobalConfig,
}

#[derive(Debug)]
pub struct InitReport {
    pub root: PathBuf,
    pub already_existed: bool,
}

impl Workspace {
    /// Walks up from `start` to the nearest directory containing `.localrag/`.
    pub fn discover(start: &Path) -> anyhow::Result<Self> {
        let start = start.canonicalize().with_context(|| format!("resolving {}", start.display()))?;
        for dir in start.ancestors() {
            if dir.join(STATE_DIR).is_dir() {
                return Self::open(dir);
            }
        }
        bail!("no localrag workspace found in {} or its parents; run `localrag init`", start.display())
    }

    pub fn open(root: &Path) -> anyhow::Result<Self> {
        if !root.join(STATE_DIR).is_dir() {
            bail!("{} is not a localrag workspace (no {STATE_DIR}/ folder)", root.display());
        }
        ensure_state_ignored(root)?;
        let config = config::load_or_default(&root.join(STATE_DIR).join("config.toml"))?;
        Ok(Workspace { root: root.to_path_buf(), config, global: config::load_global()? })
    }

    /// Creates `.localrag/` (with a default config) and `corpus/`. Safe to re-run.
    pub fn init(root: &Path) -> anyhow::Result<InitReport> {
        let root = root.canonicalize().with_context(|| format!("resolving {}", root.display()))?;
        let state = root.join(STATE_DIR);
        let already_existed = state.is_dir();
        std::fs::create_dir_all(state.join("ocr"))?;
        std::fs::create_dir_all(state.join("logs"))?;
        std::fs::create_dir_all(root.join(CORPUS_DIR))?;
        let cfg_path = state.join("config.toml");
        if !cfg_path.exists() {
            config::save(&cfg_path, &WorkspaceConfig::default(), CONFIG_HEADER)?;
        }
        ensure_state_ignored(&root)?;
        Ok(InitReport { root, already_existed })
    }

    /// Creates `parent/name` (which must not exist yet, or be an empty folder) as a new workspace.
    pub fn create(parent: &Path, name: &str) -> anyhow::Result<PathBuf> {
        let name = name.trim();
        if name.is_empty() || name.contains('/') || name.starts_with('.') || name == ".." {
            bail!("“{name}” isn't a usable folder name");
        }
        if !parent.is_dir() {
            bail!("{} is not an existing folder", parent.display());
        }
        let dir = parent.join(name);
        if dir.exists() && std::fs::read_dir(&dir)?.next().is_some() {
            bail!("{} already exists and isn't empty; open it as a workspace instead, or pick another name", dir.display());
        }
        std::fs::create_dir_all(&dir)?;
        Ok(Self::init(&dir)?.root)
    }

    pub fn save_config(&self) -> anyhow::Result<()> {
        config::save(&self.state_dir().join("config.toml"), &self.config, CONFIG_HEADER)
    }

    pub fn corpus_dir(&self) -> PathBuf {
        self.root.join(CORPUS_DIR)
    }

    pub fn state_dir(&self) -> PathBuf {
        self.root.join(STATE_DIR)
    }

    pub fn db_path(&self) -> PathBuf {
        self.state_dir().join("index.db")
    }

    pub fn ocr_cache_path(&self, sha256: &str) -> PathBuf {
        self.state_dir().join("ocr").join(format!("{sha256}.json"))
    }
}

/// `.localrag/.gitignore` and `corpus/.gitignore`, each containing `*`, make git ignore the
/// state folder (index, chat history, extracted text, settings) and the documents (often
/// copyrighted or private) in any repository, including one created after the workspace.
fn ensure_state_ignored(root: &Path) -> anyhow::Result<()> {
    const NOTE: &str = "# Written by localrag so git never commits this folder. Delete this file to version it.\n*\n";
    for (dir, note) in [
        (root.join(STATE_DIR), "# Written by localrag: nothing in .localrag/ belongs in version control\n# (index, chat history, extracted document text, settings).\n*\n"),
        (root.join(CORPUS_DIR), NOTE),
    ] {
        if !dir.is_dir() {
            continue;
        }
        let path = dir.join(".gitignore");
        if path.exists() {
            continue; // respect an existing (possibly user-edited) file
        }
        std::fs::write(&path, note).with_context(|| format!("writing {}", path.display()))?;
    }
    Ok(())
}
