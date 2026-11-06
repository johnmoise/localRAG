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
    /// The UI asked for the Crossref email once (set or skipped); don't ask again.
    pub crossref_email_prompted: bool,
    /// Cached Crossref responses younger than this many days are reused instead of asking
    /// Crossref again (§7). 0 always asks.
    pub cache_refresh_interval_days: u32,
}

impl Default for GlobalConfig {
    fn default() -> Self {
        GlobalConfig {
            ollama_host: "http://127.0.0.1:11434".into(),
            crossref_email: None,
            pdfium_lib: None,
            crossref_email_prompted: false,
            cache_refresh_interval_days: crate::metadata::crossref::DEFAULT_REFRESH_DAYS,
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
    /// When a quote isn't in its source, ask the model once to copy the supporting words
    /// exactly, then verify those (§10).
    pub repair_quotes: bool,
    /// Strict mode: a second pass checks that each cited passage supports its sentence.
    pub strict: bool,
    /// Context size per chat model (§11); filled in on first use.
    pub num_ctx: BTreeMap<String, u32>,
    /// Default reasoning model for the agent and Second Brain (§12); falls back to another
    /// installed reasoning model when this one isn't installed.
    pub reasoning_model: Option<String>,
    /// Answers the agent's `ask_corpus` calls and judges its proposals. Unset: the reasoning
    /// model itself, so Ollama doesn't swap models at every step on a 16 GB machine.
    pub agent_corpus_model: Option<String>,
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
            repair_quotes: true,
            strict: false,
            num_ctx: BTreeMap::new(),
            reasoning_model: Some(DEFAULT_REASONING_MODEL.into()),
            agent_corpus_model: None,
        }
    }
}

/// A plausible email address, trimmed. Only basic shape is checked: Crossref just uses it to
/// contact you if your requests cause problems.
pub fn validate_email(s: &str) -> anyhow::Result<String> {
    let e = s.trim();
    let ok = e.len() <= 254
        && !e.contains(char::is_whitespace)
        && e.split_once('@').is_some_and(|(local, domain)| !local.is_empty() && domain.contains('.') && !domain.starts_with('.') && !domain.ends_with('.'));
    if !ok {
        anyhow::bail!("“{e}” doesn't look like an email address");
    }
    Ok(e.to_string())
}

/// The agent's default reasoning model (§12): needs Ollama's thinking and tools capabilities.
pub const DEFAULT_REASONING_MODEL: &str = "qwen3.5:9b";

/// Upper bound for `cache_refresh_interval_days` (ten years).
pub const MAX_CACHE_REFRESH_DAYS: u32 = 3650;

pub fn validate_cache_days(s: &str) -> anyhow::Result<u32> {
    match s.trim().parse::<u32>() {
        Ok(d) if d <= MAX_CACHE_REFRESH_DAYS => Ok(d),
        _ => anyhow::bail!("cache_refresh_interval_days must be a whole number of days from 0 to {MAX_CACHE_REFRESH_DAYS}"),
    }
}

/// The user's home folder (`$HOME`; `%USERPROFILE%` on Windows).
pub fn home_dir() -> Option<PathBuf> {
    std::env::home_dir().filter(|p| !p.as_os_str().is_empty())
}

/// `$LOCALRAG_CONFIG_DIR`, else `~/.config/localrag` (`%APPDATA%\localrag` on Windows, which
/// only this user can read).
pub fn global_config_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LOCALRAG_CONFIG_DIR") {
        return PathBuf::from(dir);
    }
    if cfg!(windows)
        && let Some(appdata) = std::env::var_os("APPDATA").filter(|v| !v.is_empty())
    {
        return PathBuf::from(appdata).join("localrag");
    }
    home_dir().unwrap_or_else(|| PathBuf::from(".")).join(".config").join("localrag")
}

/// Per-user cache folder: `$LOCALRAG_CACHE_DIR`, else `~/.cache/localrag`
/// (`%LOCALAPPDATA%\localrag` on Windows).
pub fn cache_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("LOCALRAG_CACHE_DIR") {
        return PathBuf::from(dir);
    }
    if cfg!(windows)
        && let Some(local) = std::env::var_os("LOCALAPPDATA").filter(|v| !v.is_empty())
    {
        return PathBuf::from(local).join("localrag");
    }
    home_dir().unwrap_or_default().join(".cache").join("localrag")
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

/// Creates `dir` (and parents) and makes it accessible to this user only (0700). On Windows
/// the per-user folders it is used under (`%APPDATA%`, `%LOCALAPPDATA%`) are already private
/// by their ACLs, which new files and folders inherit.
pub fn ensure_private_dir(dir: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))?;
    }
    Ok(())
}

/// Options for creating a file only this user can read (0600 on Unix; the folder's ACL on
/// Windows).
pub fn private_file_options() -> std::fs::OpenOptions {
    #[allow(unused_mut)]
    let mut opts = std::fs::OpenOptions::new();
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts
}

/// Writes a file readable and writable by this user only (0600), replacing it atomically.
pub fn write_private(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    use std::io::Write;
    // A unique temp name: concurrent writers of the same file must not rename each other's.
    static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let tmp = path.with_file_name(format!(".{name}.{}-{seq}.tmp", std::process::id()));
    let mut f = private_file_options().create(true).truncate(true).write(true).open(&tmp)
        .with_context(|| format!("writing {}", tmp.display()))?;
    f.write_all(bytes)?;
    f.sync_all()?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))
}
