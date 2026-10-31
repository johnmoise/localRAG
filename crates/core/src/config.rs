//! Global (`~/.config/localrag/config.toml`) and workspace (`.localrag/config.toml`)
//! configuration (ARCHITECTURE.md §4).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct GlobalConfig {
    pub ollama_host: String,
    /// Sent to Crossref as `mailto` for the polite pool (§7). Phase 4.
    pub crossref_email: Option<String>,
    /// Overrides PDFium library discovery.
    pub pdfium_lib: Option<PathBuf>,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        GlobalConfig {
            ollama_host: "http://127.0.0.1:11434".into(),
            crossref_email: None,
            pdfium_lib: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OcrMode {
    Ask,
    Auto,
    Off,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ChunkingConfig {
    /// Target chunk size in (estimated) tokens.
    pub target_tokens: usize,
    /// Fraction of a chunk repeated at the start of the next one.
    pub overlap: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        ChunkingConfig { target_tokens: 500, overlap: 0.15 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkspaceConfig {
    pub embed_model: String,
    pub chat_model: String,
    pub ocr: OcrMode,
    /// Files larger than this are skipped.
    pub max_file_mb: u64,
    pub chunking: ChunkingConfig,
    /// Context size per chat model (§11); filled in on first use.
    pub num_ctx: BTreeMap<String, u32>,
}

impl Default for WorkspaceConfig {
    fn default() -> Self {
        WorkspaceConfig {
            embed_model: "qwen3-embedding:0.6b".into(),
            chat_model: "qwen3.5:4b".into(),
            ocr: OcrMode::Ask,
            max_file_mb: 200,
            chunking: ChunkingConfig::default(),
            num_ctx: BTreeMap::new(),
        }
    }
}

/// `$LOCALRAG_CONFIG_DIR`, else `~/.config/localrag`.
pub fn global_config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LOCALRAG_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
    home.join(".config").join("localrag")
}

pub fn load_global() -> anyhow::Result<GlobalConfig> {
    load_or_default(&global_config_dir().join("config.toml"))
}

pub(crate) fn load_or_default<T: Default + for<'de> Deserialize<'de>>(path: &Path) -> anyhow::Result<T> {
    match std::fs::read_to_string(path) {
        Ok(text) => toml::from_str(&text).with_context(|| format!("parsing {}", path.display())),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(T::default()),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

pub(crate) fn save<T: Serialize>(path: &Path, value: &T, header: &str) -> anyhow::Result<()> {
    let body = toml::to_string_pretty(value)?;
    std::fs::write(path, format!("{header}{body}")).with_context(|| format!("writing {}", path.display()))
}
