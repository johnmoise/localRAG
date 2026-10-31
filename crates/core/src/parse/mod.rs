//! Turning corpus files into pages of text blocks (ARCHITECTURE.md §5).

pub mod markdown;
pub mod pdf;
pub mod text;

use std::ops::Range;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocKind {
    Pdf,
    Markdown,
    Text,
}

impl DocKind {
    pub fn from_path(path: &Path) -> Option<Self> {
        let ext = path.extension()?.to_str()?.to_ascii_lowercase();
        match ext.as_str() {
            "pdf" => Some(DocKind::Pdf),
            "md" | "markdown" => Some(DocKind::Markdown),
            "txt" => Some(DocKind::Text),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            DocKind::Pdf => "pdf",
            DocKind::Markdown => "markdown",
            DocKind::Text => "text",
        }
    }

    /// Bump a parser's version to make the next sync reparse every file of that kind.
    pub fn parser_version(self) -> i64 {
        match self {
            DocKind::Pdf => pdf::PARSER_VERSION,
            DocKind::Markdown => markdown::PARSER_VERSION,
            DocKind::Text => text::PARSER_VERSION,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct ParsedDoc {
    pub pages: Vec<Page>,
    /// 1-based PDF pages with no text layer (§6).
    pub scanned_pages: Vec<u32>,
    pub meta: DocMeta,
}

#[derive(Debug, Clone, Default)]
pub struct DocMeta {
    pub title: Option<String>,
    pub authors: Vec<String>,
    pub year: Option<i32>,
    pub doi: Option<String>,
    /// Where these fields came from: "pdf" or "frontmatter".
    pub source: Option<&'static str>,
}

#[derive(Debug, Clone, Default)]
pub struct Page {
    /// 1-based page index for PDFs; `None` for Markdown and text, which are one "page".
    pub pdf_page: Option<u32>,
    /// Printed page number from the PDF's page labels, if it has them.
    pub label: Option<String>,
    pub text: String,
    pub blocks: Vec<Block>,
}

/// A paragraph-like run of text within one page.
#[derive(Debug, Clone)]
pub struct Block {
    /// Char offsets into `Page::text`.
    pub chars: Range<usize>,
    /// 1-based line numbers (Markdown/text only).
    pub lines: Option<Range<u32>>,
    /// Heading path, outermost first.
    pub section: Vec<String>,
    pub ocr: bool,
}

/// A failure caused by the file itself. Recorded as `failed` and retried only when the file
/// changes. Anything else (missing PDFium, I/O on our side) aborts the sync instead.
#[derive(Debug)]
pub struct FileError(pub String);

impl std::fmt::Display for FileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for FileError {}

/// Builds a page whose blocks are runs of non-blank lines, for plain text and Markdown bodies.
/// `heading` classifies a line as a heading: Some((level, title)).
pub(crate) fn line_blocks(text: &str, heading: impl Fn(&str) -> Option<(usize, String)>) -> Vec<Block> {
    let mut blocks = Vec::new();
    let mut section: Vec<String> = Vec::new();
    let mut current: Option<(usize, u32)> = None; // (char start, first line)
    let mut char_pos = 0usize;
    let mut in_fence = false;

    let close = |blocks: &mut Vec<Block>, current: &mut Option<(usize, u32)>, end_char: usize, end_line: u32, section: &[String]| {
        if let Some((start, first)) = current.take() {
            blocks.push(Block { chars: start..end_char, lines: Some(first..end_line), section: section.to_vec(), ocr: false });
        }
    };

    let mut last_content_end = 0usize;
    for (i, line) in text.split_inclusive('\n').enumerate() {
        let line_no = i as u32 + 1;
        let trimmed = line.trim();
        let line_chars = line.chars().count();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
        }
        let as_heading = if in_fence { None } else { heading(trimmed) };
        if trimmed.is_empty() || as_heading.is_some() {
            close(&mut blocks, &mut current, last_content_end, line_no, &section);
            if let Some((level, title)) = as_heading {
                section.truncate(level.saturating_sub(1));
                section.push(title);
            }
        } else {
            if current.is_none() {
                current = Some((char_pos, line_no));
            }
            last_content_end = char_pos + line.trim_end().chars().count();
        }
        char_pos += line_chars;
    }
    let total_lines = text.split_inclusive('\n').count() as u32 + 1;
    close(&mut blocks, &mut current, last_content_end, total_lines, &section);
    blocks
}
