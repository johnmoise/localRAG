//! OCR for scanned PDF pages with Apple Vision (ARCHITECTURE.md §6).
//!
//! Pages are rendered by the sandboxed PDF helper and recognized here. Results are cached in
//! `.localrag/ocr/<sha256>.json`, keyed by file content, so rechunking, a model change or a
//! rename never repeats OCR; corpus files are never modified. At indexing time the cached
//! text replaces the empty scanned pages, its blocks flagged `ocr`.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, bail};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSData, NSDictionary};
use objc2_vision::{VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel};
use serde::{Deserialize, Serialize};

use crate::parse::{ParsedDoc, line_blocks};
use crate::pdfhelper::PdfHelper;
use crate::workspace::Workspace;

/// Bump to invalidate cached OCR (e.g. a different engine or settings).
pub const ENGINE: &str = "apple-vision-accurate-1";
/// ~300 DPI for a US-letter page.
const RENDER_WIDTH: u32 = 2550;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OcrCache {
    pub engine: String,
    /// By 1-based PDF page.
    pub pages: BTreeMap<u32, OcrPage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrPage {
    /// Lines joined with "\n", a blank line between paragraphs.
    pub text: String,
    pub lines: Vec<OcrLine>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OcrLine {
    pub text: String,
    /// Char offset of this line in `OcrPage::text`.
    pub start: usize,
    /// [x, y, width, height] as fractions of the page, origin top-left.
    pub bbox: [f32; 4],
    pub confidence: f32,
}

pub fn load_cache(ws: &Workspace, sha256: &str) -> Option<OcrCache> {
    let text = std::fs::read_to_string(ws.ocr_cache_path(sha256)).ok()?;
    serde_json::from_str::<OcrCache>(&text).ok().filter(|c| c.engine == ENGINE)
}

fn save_cache(ws: &Workspace, sha256: &str, cache: &OcrCache) -> anyhow::Result<()> {
    let path = ws.ocr_cache_path(sha256);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    crate::config::write_private(&path, serde_json::to_string(cache)?.as_bytes())
}

/// OCRs `pages` of the PDF at `path`, saving after each page so an interrupted run resumes.
/// `progress(done, total)` is called as pages finish; `cancel` stops between pages.
pub fn ocr_document(
    ws: &Workspace,
    helper: &PdfHelper,
    path: &std::path::Path,
    sha256: &str,
    pages: &[u32],
    cancel: &AtomicBool,
    progress: &mut dyn FnMut(usize, usize),
) -> anyhow::Result<OcrCache> {
    let mut cache = load_cache(ws, sha256).unwrap_or(OcrCache { engine: ENGINE.into(), pages: BTreeMap::new() });
    let todo: Vec<u32> = pages.iter().copied().filter(|p| !cache.pages.contains_key(p)).collect();
    let total = todo.len();
    for (i, &page) in todo.iter().enumerate() {
        if cancel.load(Ordering::SeqCst) {
            bail!("OCR cancelled");
        }
        progress(i, total);
        let png = helper.render(path, page, RENDER_WIDTH)?.map_err(|e| anyhow::anyhow!("page {page}: {e}"))?;
        let lines = recognize(&png).with_context(|| format!("OCR of page {page}"))?;
        cache.pages.insert(page, layout(lines));
        save_cache(ws, sha256, &cache)?;
    }
    progress(total, total);
    Ok(cache)
}

/// Replaces scanned pages that have cached OCR with the recognized text.
pub fn apply_to(doc: &mut ParsedDoc, cache: &OcrCache) {
    for page in &mut doc.pages {
        let Some(n) = page.pdf_page else { continue };
        if !doc.scanned_pages.contains(&n) {
            continue;
        }
        if let Some(ocr) = cache.pages.get(&n) {
            page.text = ocr.text.clone();
            page.blocks = line_blocks(&ocr.text, |_| None)
                .into_iter()
                .map(|mut b| {
                    b.lines = None;
                    b.ocr = true;
                    b
                })
                .collect();
        }
    }
    doc.scanned_pages.retain(|p| !cache.pages.contains_key(p));
}

/// (text, confidence, bbox top-left fractions) per recognized line.
fn recognize(png: &[u8]) -> anyhow::Result<Vec<(String, f32, [f32; 4])>> {
    let data = NSData::with_bytes(png);
    let handler = VNImageRequestHandler::initWithData_options(VNImageRequestHandler::alloc(), &data, &NSDictionary::new());
    let request = VNRecognizeTextRequest::new();
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
    request.setUsesLanguageCorrection(true);
    let as_request: Retained<VNRequest> = Retained::into_super(Retained::into_super(request.clone()));
    handler
        .performRequests_error(&NSArray::from_retained_slice(&[as_request]))
        .map_err(|e| anyhow::anyhow!("Vision request failed: {e}"))?;
    let Some(results) = request.results() else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for obs in results.iter() {
        let Some(best) = obs.topCandidates(1).firstObject() else { continue };
        // Vision's box is normalized with the origin at the bottom-left.
        let r = unsafe { obs.boundingBox() };
        let (x, y, w, h) = (r.origin.x as f32, r.origin.y as f32, r.size.width as f32, r.size.height as f32);
        out.push((best.string().to_string(), best.confidence(), [x, 1.0 - y - h, w, h]));
    }
    Ok(out)
}

/// Orders lines top to bottom (then left to right) and starts a new paragraph where the gap
/// to the previous line is clearly larger than a line.
fn layout(mut lines: Vec<(String, f32, [f32; 4])>) -> OcrPage {
    lines.sort_by(|a, b| {
        let row = |l: &(String, f32, [f32; 4])| (l.2[1] / l.2[3].max(0.005)).round();
        row(a).total_cmp(&row(b)).then(a.2[0].total_cmp(&b.2[0]))
    });
    let mut text = String::new();
    let mut out = Vec::new();
    let mut prev: Option<[f32; 4]> = None;
    for (line, confidence, bbox) in lines {
        if let Some(p) = prev {
            let gap = bbox[1] - (p[1] + p[3]);
            text.push_str(if gap > 1.0 * p[3].max(bbox[3]) { "\n\n" } else { "\n" });
        }
        out.push(OcrLine { start: text.chars().count(), text: line.clone(), bbox, confidence });
        text.push_str(&line);
        prev = Some(bbox);
    }
    OcrPage { text, lines: out }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_orders_lines_and_marks_paragraphs() {
        let page = layout(vec![
            ("second line".into(), 0.9, [0.1, 0.13, 0.5, 0.02]),
            ("first line".into(), 0.9, [0.1, 0.10, 0.5, 0.02]),
            ("new paragraph".into(), 0.9, [0.1, 0.25, 0.5, 0.02]),
        ]);
        assert_eq!(page.text, "first line\nsecond line\n\nnew paragraph");
        assert_eq!(page.lines[1].start, "first line\n".len());
    }
}
