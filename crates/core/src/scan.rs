//! Walking `corpus/` and fingerprinting files (ARCHITECTURE.md §5).

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use anyhow::Context;
use sha2::{Digest, Sha256};

use crate::parse::DocKind;
use crate::workspace::IGNORE_FILE;

#[derive(Debug, Clone)]
pub struct CorpusFile {
    /// Relative to `corpus/`, '/'-separated.
    pub rel_path: String,
    pub abs_path: PathBuf,
    pub kind: DocKind,
    pub size: u64,
    pub mtime_ns: i64,
}

#[derive(Debug, Default)]
pub struct ScanResult {
    pub files: Vec<CorpusFile>,
    /// Supported files skipped for exceeding the size cap: (path, size in bytes).
    pub too_large: Vec<(String, u64)>,
}

/// Lists supported files under `corpus/`, skipping hidden files and `.localragignore` matches.
pub fn scan(corpus: &Path, max_bytes: u64) -> anyhow::Result<ScanResult> {
    let mut result = ScanResult::default();
    if !corpus.is_dir() {
        return Ok(result);
    }
    let walker = ignore::WalkBuilder::new(corpus)
        .hidden(true)
        .git_ignore(false)
        .git_global(false)
        .git_exclude(false)
        .ignore(false)
        .parents(false)
        .add_custom_ignore_filename(IGNORE_FILE)
        .sort_by_file_path(|a, b| a.cmp(b))
        .build();
    for entry in walker {
        let entry = entry?;
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        let Some(kind) = DocKind::from_path(path) else { continue };
        let meta = entry.metadata().with_context(|| format!("stat {}", path.display()))?;
        let rel_path = rel_path(corpus, path);
        if meta.len() > max_bytes {
            result.too_large.push((rel_path, meta.len()));
            continue;
        }
        let mtime_ns = meta
            .modified()?
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as i64)
            .unwrap_or(0);
        result.files.push(CorpusFile { rel_path, abs_path: path.to_path_buf(), kind, size: meta.len(), mtime_ns });
    }
    Ok(result)
}

fn rel_path(corpus: &Path, path: &Path) -> String {
    path.strip_prefix(corpus)
        .unwrap_or(path)
        .components()
        .map(|c| c.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

pub fn sha256_file(path: &Path) -> anyhow::Result<String> {
    let mut file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hex(&hasher.finalize()))
}

pub fn sha256_str(text: &str) -> String {
    hex(&Sha256::digest(text.as_bytes()))
}

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}
