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
pub struct RetrievalConfig {
    /// Below this best cosine similarity the model isn't called: "not found" (§10). Calibrate with `localrag eval`.
    pub min_similarity: f32,
    /// Most sources given to the model per question.
    pub max_sources: usize,
    /// Candidates from each of BM25 and vector search.
    pub candidates: usize,
    pub per_doc_cap: usize,
    /// How many top hits get their neighbouring chunks attached.
    pub expand_top: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        RetrievalConfig { min_similarity: 0.45, max_sources: 10, candidates: 30, per_doc_cap: 3, expand_top: 2 }
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
    pub retrieval: RetrievalConfig,
    /// Pass `think` to thinking-capable models when answering.
    pub think: bool,
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
            retrieval: RetrievalConfig::default(),
            think: false,
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
    write_private(path, format!("{header}{body}").as_bytes())
}

pub fn save_global(cfg: &GlobalConfig) -> anyhow::Result<()> {
    let dir = global_config_dir();
    ensure_private_dir(&dir)?;
    save(&dir.join("config.toml"), cfg, "# localrag machine-wide settings. Private to this user (mode 0600); never inside a workspace.\n\n")
}

/// Creates `dir` (and parents) and makes it accessible to this user only (0700).
pub fn ensure_private_dir(dir: &Path) -> anyhow::Result<()> {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    Ok(())
}

/// Writes a file readable and writable by this user only (0600), replacing it atomically.
pub fn write_private(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let tmp = path.with_extension("tmp");
    let mut f = std::fs::OpenOptions::new().create(true).truncate(true).write(true).mode(0o600).open(&tmp)
        .with_context(|| format!("writing {}", tmp.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}
