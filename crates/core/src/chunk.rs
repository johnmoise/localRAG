//! Structure-aware chunking (ARCHITECTURE.md §5): sentences packed to a token target with
//! overlap, never crossing a section boundary and never splitting a sentence (unless the
//! sentence alone exceeds the target).

use unicode_segmentation::UnicodeSegmentation;

use crate::config::ChunkingConfig;
use crate::parse::ParsedDoc;
use crate::scan::sha256_str;

/// Bump to make the next sync rechunk every document.
pub const CHUNKER_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct Chunk {
    pub ord: usize,
    pub text: String,
    pub text_sha: String,
    /// Indexes into `ParsedDoc::pages`.
    pub page_start: usize,
    pub page_end: usize,
    /// Char offset into the text of `page_start`, and into the text of `page_end`.
    pub char_start: usize,
    pub char_end: usize,
    pub lines: Option<(u32, u32)>,
    pub section: String,
    pub n_tokens: usize,
    pub ocr: bool,
}

/// Rough token count: about 4 characters per token for English prose.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4).max(1)
}

#[derive(Debug, Clone)]
struct Unit {
    text: String,
    page: usize,
    char_start: usize,
    char_end: usize,
    lines: Option<(u32, u32)>,
    section: String,
    tokens: usize,
    ocr: bool,
}

pub fn chunk(doc: &ParsedDoc, cfg: &ChunkingConfig) -> Vec<Chunk> {
    let target = cfg.target_tokens.max(32);
    let overlap = ((target as f32) * cfg.overlap.clamp(0.0, 0.5)) as usize;
    let units = units(doc, target);

    let mut chunks = Vec::new();
    let mut current: Vec<&Unit> = Vec::new();
    let mut tokens = 0usize;
    for unit in &units {
        let section_changed = current.last().is_some_and(|u| u.section != unit.section);
        if !current.is_empty() && (section_changed || tokens + unit.tokens > target) {
            chunks.push(build(chunks.len(), &current));
            // Carry trailing sentences into the next chunk, unless the section changed.
            let mut carried: Vec<&Unit> = Vec::new();
            if !section_changed {
                let mut carried_tokens = 0;
                for u in current.iter().rev() {
                    if carried_tokens + u.tokens > overlap || carried.len() + 1 == current.len() {
                        break;
                    }
                    carried_tokens += u.tokens;
                    carried.push(u);
                }
                carried.reverse();
            }
            tokens = carried.iter().map(|u| u.tokens).sum();
            current = carried;
        }
        tokens += unit.tokens;
        current.push(unit);
    }
    if !current.is_empty() {
        chunks.push(build(chunks.len(), &current));
    }
    chunks
}

fn build(ord: usize, units: &[&Unit]) -> Chunk {
    let first = units[0];
    let last = units[units.len() - 1];
    let text = units.iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join(" ");
    let lines = match (first.lines, last.lines) {
        (Some((a, _)), Some((_, b))) => Some((a, b)),
        _ => None,
    };
    Chunk {
        ord,
        text_sha: sha256_str(&text),
        n_tokens: estimate_tokens(&text),
        text,
        page_start: first.page,
        page_end: last.page,
        char_start: first.char_start,
        char_end: last.char_end,
        lines,
        section: first.section.clone(),
        ocr: units.iter().any(|u| u.ocr),
    }
}

/// Sentences of every block, in reading order. Sentences longer than `target` are split
/// into word runs.
fn units(doc: &ParsedDoc, target: usize) -> Vec<Unit> {
    let mut out = Vec::new();
    for (page_idx, page) in doc.pages.iter().enumerate() {
        let chars: Vec<(usize, char)> = page.text.char_indices().collect();
        let byte_at = |c: usize| chars.get(c).map_or(page.text.len(), |&(b, _)| b);
        for block in &page.blocks {
            let block_text = &page.text[byte_at(block.chars.start)..byte_at(block.chars.end)];
            let section = block.section.join(" > ");
            for (byte_off, sentence) in block_text.split_sentence_bound_indices() {
                let text = normalize(sentence);
                if text.is_empty() {
                    continue;
                }
                let lead = sentence.chars().take_while(|c| c.is_whitespace()).count();
                let start = block.chars.start + block_text[..byte_off].chars().count() + lead;
                let end = start + sentence.trim().chars().count();
                let lines = block.lines.as_ref().map(|r| {
                    let line_of = |c: usize| r.start + page.text.chars().skip(block.chars.start).take(c - block.chars.start).filter(|&ch| ch == '\n').count() as u32;
                    (line_of(start), line_of(end.max(start + 1) - 1))
                });
                let base = Unit { text: String::new(), page: page_idx, char_start: start, char_end: end, lines, section: section.clone(), tokens: 0, ocr: block.ocr };
                for piece in split_long(&text, target) {
                    out.push(Unit { tokens: estimate_tokens(&piece), text: piece, ..base.clone() });
                }
            }
        }
    }
    out
}

/// Joins wrapped lines: drops end-of-line hyphenation before a lowercase letter and
/// collapses all whitespace to single spaces.
fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.trim().chars().peekable();
    while let Some(c) = chars.next() {
        if c == '-' && chars.peek() == Some(&'\n') {
            let mut look = chars.clone();
            look.next();
            if look.peek().is_some_and(|n| n.is_lowercase()) {
                chars.next();
                continue;
            }
        }
        if c.is_whitespace() {
            if !out.ends_with(' ') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn split_long(text: &str, target: usize) -> Vec<String> {
    if estimate_tokens(text) <= target {
        return vec![text.to_string()];
    }
    let mut pieces = Vec::new();
    let mut piece = String::new();
    for word in text.split(' ') {
        if !piece.is_empty() && estimate_tokens(&piece) + estimate_tokens(word) > target {
            pieces.push(std::mem::take(&mut piece));
        }
        if !piece.is_empty() {
            piece.push(' ');
        }
        piece.push_str(word);
    }
    if !piece.is_empty() {
        pieces.push(piece);
    }
    pieces
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{markdown, text};

    fn cfg(target: usize) -> ChunkingConfig {
        ChunkingConfig { target_tokens: target, overlap: 0.15 }
    }

    #[test]
    fn packs_sentences_with_overlap_and_respects_sections() {
        let sentence = "This sentence has roughly forty characters. ";
        let md = format!("# A\n{}\n\n# B\nShort section.\n", sentence.repeat(20));
        let doc = markdown::parse(md.as_bytes()).unwrap();
        let chunks = chunk(&doc, &cfg(50));
        assert!(chunks.len() >= 4, "{} chunks", chunks.len());
        assert!(chunks.iter().all(|c| c.n_tokens <= 60));
        // Overlap: the next chunk starts with the previous chunk's last sentence.
        let last_sentence = chunks[0].text.rsplit(". ").next().unwrap();
        assert!(chunks[1].text.starts_with(last_sentence.trim_end_matches('.')));
        // The new section starts a fresh chunk with no overlap.
        let b = chunks.iter().find(|c| c.section == "B").unwrap();
        assert_eq!(b.text, "Short section.");
        assert_eq!(b.lines, Some((5, 5)));
    }

    #[test]
    fn offsets_point_back_into_the_page() {
        let txt = "First para line one\nline two.\n\nSecond para.\n";
        let doc = text::parse(txt.as_bytes()).unwrap();
        let chunks = chunk(&doc, &cfg(500));
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert_eq!(c.text, "First para line one line two. Second para.");
        let page: Vec<char> = doc.pages[0].text.chars().collect();
        let span: String = page[c.char_start..c.char_end].iter().collect();
        assert_eq!(span, "First para line one\nline two.\n\nSecond para.");
        assert_eq!(c.lines, Some((1, 4)));
    }

    #[test]
    fn dehyphenates_wrapped_words() {
        assert_eq!(normalize("quad-\nratic time"), "quadratic time");
        assert_eq!(normalize("Self-\nAttention"), "Self- Attention");
    }
}
