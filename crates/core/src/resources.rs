//! Locating bundled files (PDFium, tokenizers) in an app bundle or a dev checkout.

use std::path::{Path, PathBuf};

/// Looks for `rel` (e.g. "tokenizers/qwen35.json") next to the executable, in the app
/// bundle's Resources/Frameworks folders, and under `third_party/` in any ancestor of the
/// executable or the current directory (a dev checkout).
pub fn find(rel: &str) -> Option<PathBuf> {
    let name = Path::new(rel).file_name()?;
    let mut candidates = Vec::new();
    if let Some(dir) = std::env::current_exe().ok().and_then(|e| e.parent().map(Path::to_path_buf)) {
        candidates.push(dir.join(name));
        candidates.push(dir.join("../Resources").join(rel));
        candidates.push(dir.join("../Frameworks").join(name));
        candidates.extend(dir.ancestors().map(|a| a.join("third_party").join(rel)));
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("third_party").join(rel));
    }
    candidates.into_iter().find(|p| p.is_file())
}
