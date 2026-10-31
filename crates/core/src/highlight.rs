//! Mapping a span of chunk text back to the source file: rectangles on PDF pages, or char
//! offsets in a text/Markdown file (ARCHITECTURE.md §10).

use anyhow::Context;
use pdfium_render::prelude::*;
use rusqlite::Connection;
use serde::Serialize;

use crate::chunk::normalize_mapped;
use crate::scan;
use crate::workspace::Workspace;

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Highlight {
    /// Rectangles as fractions of the page (origin top-left): [x, y, width, height].
    Pdf { pages: Vec<PageRects> },
    /// Char offsets into the file's text.
    Text { start: usize, end: usize },
    /// The file is gone or changed since indexing; only the chunk text can be shown.
    Unavailable { reason: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct PageRects {
    /// 1-based.
    pub pdf_page: u32,
    pub rects: Vec<[f32; 4]>,
}

struct ChunkLoc {
    path: String,
    doc_sha: String,
    kind: String,
    status: String,
    text: String,
    page_start: Option<u32>,
    page_end: Option<u32>,
    char_start: usize,
    char_end: usize,
}

/// Highlights `span` (char range in the chunk's text), or the whole chunk when `None`.
pub fn for_chunk(ws: &Workspace, conn: &Connection, chunk_id: i64, span: Option<(usize, usize)>) -> anyhow::Result<Highlight> {
    let loc: ChunkLoc = conn
        .query_row(
            "SELECT d.path, d.sha256, d.kind, d.status, c.text, c.pdf_page_start, c.pdf_page_end, c.char_start, c.char_end
             FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.id = ?1",
            [chunk_id],
            |r| {
                Ok(ChunkLoc {
                    path: r.get(0)?,
                    doc_sha: r.get(1)?,
                    kind: r.get(2)?,
                    status: r.get(3)?,
                    text: r.get(4)?,
                    page_start: r.get(5)?,
                    page_end: r.get(6)?,
                    char_start: r.get::<_, i64>(7)? as usize,
                    char_end: r.get::<_, i64>(8)? as usize,
                })
            },
        )
        .with_context(|| format!("no chunk {chunk_id}"))?;
    if loc.status == "removed" {
        return Ok(Highlight::Unavailable { reason: "the file was removed from corpus/".into() });
    }
    let file = ws.corpus_dir().join(&loc.path);
    if !file.is_file() || scan::sha256_file(&file)? != loc.doc_sha {
        return Ok(Highlight::Unavailable { reason: "the file changed since it was indexed; reindex to update".into() });
    }
    let target: Vec<char> = match span {
        Some((s, e)) => loc.text.chars().skip(s).take(e.saturating_sub(s)).collect(),
        None => loc.text.chars().collect(),
    };
    let (target, _) = normalize_mapped(&target);

    if loc.kind == "pdf" {
        let lib = crate::parse::pdf::find_library(ws.global.pdfium_lib.as_deref())?;
        let pdfium = crate::parse::pdf::pdfium(&lib)?;
        pdf_highlight(pdfium, &std::fs::read(&file)?, &loc, &target)
    } else {
        let text = std::fs::read_to_string(&file)?;
        let page: Vec<char> = text.chars().collect();
        let region = &page[loc.char_start.min(page.len())..loc.char_end.min(page.len())];
        let (norm, map) = normalize_mapped(region);
        Ok(match find(&norm, &map, &target) {
            Some((a, b)) => Highlight::Text { start: loc.char_start + a, end: loc.char_start + b + 1 },
            None => Highlight::Text { start: loc.char_start, end: loc.char_end },
        })
    }
}

/// (first, last) region indices of `target` within the normalized region.
fn find(norm: &str, map: &[usize], target: &str) -> Option<(usize, usize)> {
    if target.is_empty() {
        return None;
    }
    let byte = norm.find(target)?;
    let first = norm[..byte].chars().count();
    let last = first + target.chars().count() - 1;
    Some((map[first], map[last]))
}

fn pdf_highlight(pdfium: &Pdfium, bytes: &[u8], loc: &ChunkLoc, target: &str) -> anyhow::Result<Highlight> {
    let doc = pdfium.load_pdf_from_byte_slice(bytes, None)?;
    let (first, last) = (loc.page_start.unwrap_or(1), loc.page_end.or(loc.page_start).unwrap_or(1));

    // The chunk's region, as (page, char index, char) across its pages. Our stored offsets
    // count "\r\n" as one char (the parser normalizes it), PDFium's count it as two.
    let mut region: Vec<(u32, usize, char)> = Vec::new();
    let mut boxes: Vec<Vec<Option<PdfRect>>> = Vec::new();
    let mut sizes = Vec::new();
    for p in first..=last {
        let page = doc.pages().get((p - 1) as i32)?;
        sizes.push((page.width().value, page.height().value));
        let text = page.text()?;
        let chars: Vec<(char, Option<PdfRect>)> =
            text.chars().iter().map(|c| (c.unicode_char().unwrap_or(' '), c.loose_bounds().ok())).collect();
        boxes.push(chars.iter().map(|c| c.1).collect());
        let mut ours = 0usize; // index in the parser's text (with "\r\n" → "\n")
        for (i, (c, _)) in chars.iter().enumerate() {
            if *c == '\r' && chars.get(i + 1).is_some_and(|n| n.0 == '\n') {
                continue;
            }
            let in_region = (p > first || ours >= loc.char_start) && (p < last || ours < loc.char_end);
            if in_region {
                region.push((p, i, *c));
            }
            ours += 1;
        }
    }
    let chars: Vec<char> = region.iter().map(|r| r.2).collect();
    let (norm, map) = normalize_mapped(&chars);
    let (a, b) = find(&norm, &map, target).unwrap_or((0, region.len().saturating_sub(1)));

    let mut pages: Vec<PageRects> = Vec::new();
    for &(p, i, _) in &region[a..=b.min(region.len().saturating_sub(1))] {
        let Some(rect) = boxes[(p - first) as usize][i] else { continue };
        let (w, h) = sizes[(p - first) as usize];
        if pages.last().is_none_or(|pr| pr.pdf_page != p) {
            pages.push(PageRects { pdf_page: p, rects: Vec::new() });
        }
        let rects = &mut pages.last_mut().expect("pushed above").rects;
        let r = [rect.left().value / w, (h - rect.top().value) / h, (rect.right().value - rect.left().value) / w, (rect.top().value - rect.bottom().value) / h];
        // Extend the previous rect when the char continues the same line.
        if let Some(prev) = rects.last_mut()
            && (prev[1] - r[1]).abs() < r[3].max(prev[3]) * 0.5
            && r[0] >= prev[0] - 0.002
            && r[0] - (prev[0] + prev[2]) < 0.05
        {
            let right = (prev[0] + prev[2]).max(r[0] + r[2]);
            let top = prev[1].min(r[1]);
            let bottom = (prev[1] + prev[3]).max(r[1] + r[3]);
            *prev = [prev[0], top, right - prev[0], bottom - top];
        } else if r[2] > 0.0 && r[3] > 0.0 {
            rects.push(r);
        }
    }
    Ok(Highlight::Pdf { pages })
}
