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
    /// Set when the workspace is inside a git repository whose .gitignore doesn't cover `.localrag/`.
    pub gitignore_suggestion: Option<PathBuf>,
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
        Ok(InitReport { gitignore_suggestion: gitignore_suggestion(&root), root, already_existed })
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

/// Appends `.localrag/` to the enclosing repository's `.gitignore`.
pub fn add_to_gitignore(gitignore: &Path) -> anyhow::Result<()> {
    let mut text = std::fs::read_to_string(gitignore).unwrap_or_default();
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(".localrag/\n");
    std::fs::write(gitignore, text).with_context(|| format!("writing {}", gitignore.display()))
}

fn gitignore_suggestion(root: &Path) -> Option<PathBuf> {
    let repo = root.ancestors().find(|d| d.join(".git").exists())?;
    let gitignore = repo.join(".gitignore");
    let covered = std::fs::read_to_string(&gitignore)
        .map(|t| t.lines().any(|l| matches!(l.trim(), ".localrag" | ".localrag/" | "**/.localrag/" | "**/.localrag")))
        .unwrap_or(false);
    (!covered).then_some(gitignore)
}
