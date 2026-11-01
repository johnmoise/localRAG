//! Watches the active workspace's `corpus/` and syncs after changes settle (ARCHITECTURE.md
//! §5): adding, editing, removing or renaming files in Finder shows up without pressing
//! Reindex. Hidden files are ignored, including our own upload temp files and `.gitignore`.

use std::path::Path;
use std::time::Duration;

use notify_debouncer_mini::notify::{RecommendedWatcher, RecursiveMode};
use notify_debouncer_mini::{DebounceEventResult, Debouncer, new_debouncer};

use crate::AppState;

pub type Watcher = Debouncer<RecommendedWatcher>;

/// How long the corpus must be quiet before a sync starts (copying a big PDF takes a moment).
const SETTLE: Duration = Duration::from_millis(1500);

pub fn start(state: &AppState, corpus: &Path) -> anyhow::Result<Watcher> {
    let state = state.clone();
    let root = corpus.parent().map(Path::to_path_buf);
    // Events arrive with canonical paths (/private/var/… for /var/…).
    let corpus_canon = corpus.canonicalize().unwrap_or_else(|_| corpus.to_path_buf());
    let corpus_given = corpus.to_path_buf();
    let mut debouncer = new_debouncer(SETTLE, move |res: DebounceEventResult| {
        let Ok(events) = res else { return };
        let relevant = events.iter().any(|e| {
            let rel = e.path.strip_prefix(&corpus_canon).or_else(|_| e.path.strip_prefix(&corpus_given));
            rel.is_ok_and(|r| !is_hidden(r))
        });
        // Only for the workspace still active (a late event after switching is dropped).
        if relevant && state.root() == root {
            state.start_sync();
        }
    })?;
    debouncer.watcher().watch(corpus, RecursiveMode::Recursive)?;
    Ok(debouncer)
}

/// `path` is relative to corpus/: a workspace that itself lives in a hidden folder
/// (`~/.research/topic`) still gets its files watched.
fn is_hidden(path: &Path) -> bool {
    path.components().any(|c| c.as_os_str().to_string_lossy().starts_with('.'))
}

#[cfg(test)]
mod tests {
    use super::is_hidden;
    use std::path::Path;

    #[test]
    fn hidden_paths() {
        assert!(is_hidden(Path::new(".upload-1-0.tmp")));
        assert!(is_hidden(Path::new(".gitignore")));
        assert!(is_hidden(Path::new("drafts/.DS_Store")));
        assert!(!is_hidden(Path::new("paper.pdf")));
    }
}
