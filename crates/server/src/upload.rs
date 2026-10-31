//! Adding documents to the active workspace's corpus from the Library tab.
//!
//! The file is parsed exactly as indexing will parse it *before* it's written to corpus/, so
//! unsupported or broken files are refused with a specific reason instead of landing in the
//! Library as `failed`.

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Query, State};
use axum::http::StatusCode;
use serde::Deserialize;
use serde_json::json;

use localrag_core::parse::{self, DocKind};

use crate::AppState;
use crate::routes::{ApiError, ApiResult, blocking};

/// Hard ceiling on request bodies; the workspace's `max_file_mb` is checked too.
pub const MAX_UPLOAD_BYTES: usize = 1024 * 1024 * 1024;

const ACCEPTED: &str = "PDF (.pdf), Markdown (.md, .markdown) and plain text (.txt)";

#[derive(Deserialize)]
pub struct UploadQuery {
    name: String,
}

pub async fn upload(State(state): State<AppState>, Query(q): Query<UploadQuery>, body: Bytes) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let name = safe_name(&q.name)?;
        let Some(kind) = DocKind::from_path(std::path::Path::new(&name)) else {
            return Err(ApiError::new(
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "unsupported_type",
                format!("“{name}” isn't a supported file type. localrag indexes {ACCEPTED}."),
            ));
        };
        let max = ws.config.max_file_mb * 1024 * 1024;
        if body.len() as u64 > max {
            return Err(ApiError::new(
                StatusCode::PAYLOAD_TOO_LARGE,
                "too_large",
                format!("“{name}” is {:.1} MB; this workspace accepts files up to {} MB (max_file_mb in .localrag/config.toml).", body.len() as f64 / 1e6, ws.config.max_file_mb),
            ));
        }
        let pdfium = match kind {
            DocKind::Pdf => {
                let lib = parse::pdf::find_library(ws.global.pdfium_lib.as_deref())?;
                Some(parse::pdf::pdfium(&lib)?)
            }
            _ => None,
        };
        let check = parse::check_upload(kind, &body, pdfium).map_err(|e| {
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "parse_failed", format!("“{name}” couldn't be read: {e}."))
        })?;

        // Write under a hidden temporary name (the scanner skips hidden files), then rename,
        // so a sync never sees a half-written file.
        let corpus = ws.corpus_dir();
        std::fs::create_dir_all(&corpus).map_err(anyhow::Error::from)?;
        let target = unique_path(&corpus, &name);
        static SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let tmp = corpus.join(format!(".upload-{}-{seq}.tmp", std::process::id()));
        std::fs::write(&tmp, &body).map_err(anyhow::Error::from)?;
        std::fs::rename(&tmp, &target).map_err(anyhow::Error::from)?;
        state.start_sync();

        let saved_as = target.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or(name.clone());
        Ok(Json(json!({ "saved_as": saved_as, "renamed": saved_as != name, "check": check })))
    })
    .await
}

/// The file name alone: no folders, no leading dot, no control characters.
fn safe_name(raw: &str) -> Result<String, ApiError> {
    let base = raw.rsplit(['/', '\\']).next().unwrap_or("").trim();
    let clean: String = base.chars().filter(|c| !c.is_control()).collect();
    if clean.is_empty() || clean.starts_with('.') || clean == ".." {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "bad_name", format!("“{raw}” isn't a usable file name.")));
    }
    Ok(clean)
}

/// `name.pdf`, or `name (2).pdf`, `name (3).pdf`… if taken.
fn unique_path(dir: &std::path::Path, name: &str) -> std::path::PathBuf {
    let first = dir.join(name);
    if !first.exists() {
        return first;
    }
    let (stem, ext) = name.rsplit_once('.').map_or((name, ""), |(s, e)| (s, e));
    (2..)
        .map(|n| dir.join(if ext.is_empty() { format!("{stem} ({n})") } else { format!("{stem} ({n}).{ext}") }))
        .find(|p| !p.exists())
        .expect("some free name")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_reduced_to_a_plain_file_name() {
        assert_eq!(safe_name("../../etc/passwd.txt").ok().as_deref(), Some("passwd.txt"));
        assert_eq!(safe_name("C:\\Users\\me\\paper.pdf").ok().as_deref(), Some("paper.pdf"));
        assert!(safe_name(".hidden.md").is_err());
        assert!(safe_name("dir/").is_err());
    }

    #[test]
    fn collisions_get_a_number() {
        let d = tempfile::tempdir().unwrap();
        std::fs::write(d.path().join("a.pdf"), "x").unwrap();
        std::fs::write(d.path().join("a (2).pdf"), "x").unwrap();
        assert_eq!(unique_path(d.path(), "a.pdf").file_name().unwrap(), "a (3).pdf");
        assert_eq!(unique_path(d.path(), "b.pdf").file_name().unwrap(), "b.pdf");
    }
}
