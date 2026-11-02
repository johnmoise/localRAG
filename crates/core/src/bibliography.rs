//! Separating a document's bibliography from its body (docs/ARCHITECTURE.md §5).
//!
//! The body is chunked and embedded for search and answers. The bibliography is not: its
//! entries become references (enriched from Crossref by `references`), because reference
//! lists match searches without teaching anything and would crowd out real passages.

use std::sync::LazyLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::parse::{Block, ParsedDoc};

/// Where the bibliography is: (page index, char offset in that page's text), start inclusive,
/// end exclusive. Everything in between is removed from the body.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Bibliography {
    pub start: (usize, usize),
    pub end: (usize, usize),
    pub entries: Vec<RawReference>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawReference {
    /// 0-based position in the list.
    pub ord: usize,
    /// The printed number, for numbered styles ("12." or "[12]").
    pub number: Option<u32>,
    /// The entry, whitespace-normalized.
    pub text: String,
    pub doi: Option<String>,
    pub year: Option<i32>,
}

/// Bump when detection or splitting changes, to re-split every document.
pub const VERSION: i64 = 1;

/// A bibliography must start after this share of the document (a "References" line in a
/// table of contents or the abstract isn't the bibliography).
const MIN_POSITION: f64 = 0.35;

static HEADING: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:#{1,6}\s*)?(?:[0-9]{1,2}\.?\s+|[IVX]{1,4}\.\s+)?(references|bibliography|literature cited|works cited|references and notes|cited literature|reference list)\s*:?\s*$").unwrap()
});
/// What follows a bibliography and must stay in the body: tables, figures, appendices,
/// supplements; or publisher boilerplate, which belongs to neither.
static END: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)^\s*(?:(?:table|figure|fig\.)\s*[0-9S]+|appendix|appendices|supplementary|supplemental|disclaimer|publisher.s note|#{1,6}\s+\S)").unwrap()
});
/// "12. " or "[12] ". The preceding whitespace is checked separately, not matched: matching
/// it would let a previous candidate ("…4004–\n11. ") swallow the space before "16.".
static NUMBERED: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[?([0-9]{1,3})[.\]]\s+").unwrap());
static YEAR: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\b(1[89][0-9]{2}|20[0-9]{2})[a-z]?\b").unwrap());
/// Author-year styles: a new entry starts with "Surname, X" or "Surname X," on a new line.
static AUTHOR_START: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^[A-Z][\p{L}'’\-]+(?:,\s*[A-Z]|\s[A-Z]{1,3}[,.])").unwrap());
static NOISE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?i)downloaded from \S+(?:\s+by guest on [^.]*?(?:19|20)[0-9]{2})?|\[(?:crossref|pubmed|google scholar|green version)\]").unwrap()
});

/// Finds the bibliography, if the document has one.
pub fn detect(doc: &ParsedDoc) -> Option<Bibliography> {
    // Every line with its (page, char offset).
    let mut lines: Vec<(usize, usize, &str)> = Vec::new();
    for (p, page) in doc.pages.iter().enumerate() {
        let mut offset = 0;
        for line in page.text.split_inclusive('\n') {
            lines.push((p, offset, line.trim_end_matches(['\n', '\r'])));
            offset += line.chars().count();
        }
    }
    let total: usize = doc.pages.iter().map(|p| p.text.chars().count()).sum();
    if total == 0 {
        return None;
    }
    let page_starts: Vec<usize> = doc.pages.iter().scan(0, |acc, p| {
        let s = *acc;
        *acc += p.text.chars().count();
        Some(s)
    }).collect();
    let position = |i: usize| (page_starts[lines[i].0] + lines[i].1) as f64 / total as f64;

    let candidates: Vec<usize> = (0..lines.len()).filter(|&i| HEADING.is_match(lines[i].2) && position(i) >= MIN_POSITION).collect();
    // Prefer a heading followed by something that looks like an entry; else the last one.
    let looks_like_entry = |i: usize| {
        lines.iter().skip(i + 1).map(|l| l.2.trim()).find(|l| !l.is_empty()).is_some_and(|l| NUMBERED.is_match(&format!(" {l}")) || AUTHOR_START.is_match(l))
    };
    let heading = candidates.iter().copied().find(|&i| looks_like_entry(i)).or(candidates.last().copied())?;

    let end_line = (heading + 1..lines.len()).find(|&i| END.is_match(lines[i].2)).unwrap_or(lines.len());
    let start = (lines[heading].0, lines[heading].1);
    let end = match lines.get(end_line) {
        Some(l) => (l.0, l.1),
        None => (doc.pages.len() - 1, doc.pages.last().map_or(0, |p| p.text.chars().count())),
    };
    let body: Vec<&str> = lines[heading + 1..end_line].iter().map(|l| l.2).collect();
    let entries = split_entries(&body);
    if entries.is_empty() {
        return None;
    }
    Some(Bibliography { start, end, entries })
}

/// Splits the bibliography's lines into entries: numbered ("1.", "[1]") when the numbers run
/// 1, 2, 3…; otherwise author-year entries starting with a surname on a new line.
fn split_entries(lines: &[&str]) -> Vec<RawReference> {
    let text = dehyphenate(&lines.join("\n"));
    let mut starts: Vec<(usize, u32)> = Vec::new();
    let mut expect = 1u32;
    for m in NUMBERED.captures_iter(&text) {
        let at = m.get(0).expect("match").start();
        if at > 0 && !text[..at].ends_with(char::is_whitespace) {
            continue;
        }
        let n: u32 = m[1].parse().unwrap_or(0);
        // Tolerate one number lost to extraction (it merges into the previous entry).
        if n == expect || (n == expect + 1 && expect > 1) {
            starts.push((at, n));
            expect = n + 1;
        }
    }
    let pieces: Vec<(Option<u32>, String)> = if starts.len() >= 3 {
        starts
            .iter()
            .enumerate()
            .map(|(i, &(s, n))| {
                let e = starts.get(i + 1).map_or(text.len(), |x| x.0);
                let body = NUMBERED.replace(&text[s..e], " ").to_string();
                (Some(n), body)
            })
            .collect()
    } else {
        let mut out: Vec<String> = Vec::new();
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let prev_done = out.last().is_none_or(|p| p.ends_with('.') || p.ends_with(')'));
            if out.is_empty() || (prev_done && AUTHOR_START.is_match(line)) {
                out.push(line.to_string());
            } else if let Some(last) = out.last_mut() {
                last.push(' ');
                last.push_str(line);
            }
        }
        out.into_iter().map(|t| (None, t)).collect()
    };
    pieces
        .into_iter()
        .map(|(n, t)| (n, clean(&t)))
        .filter(|(_, t)| t.chars().filter(|c| c.is_alphabetic()).count() >= 15)
        .enumerate()
        .map(|(ord, (number, text))| RawReference {
            ord,
            number,
            doi: crate::metadata::find_doi(&text),
            year: YEAR.captures(&text).and_then(|c| c[1].parse().ok()),
            text,
        })
        .collect()
}

fn dehyphenate(s: &str) -> String {
    static WRAP: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(\p{Ll})[-\u{2}]\n(\p{Ll})").unwrap());
    WRAP.replace_all(s, "$1$2").to_string()
}

fn clean(s: &str) -> String {
    let s = NOISE.replace_all(s, " ");
    s.split_whitespace().collect::<Vec<_>>().join(" ").trim_matches([' ', ';']).to_string()
}

/// Removes the bibliography from the body: blocks inside it are dropped, blocks straddling
/// its edges are cut. Page text is kept, so char offsets (and highlighting) still line up.
pub fn remove_from_body(doc: &mut ParsedDoc, bib: &Bibliography) {
    for (p, page) in doc.pages.iter_mut().enumerate() {
        if p < bib.start.0 || p > bib.end.0 {
            continue;
        }
        let lo = if p == bib.start.0 { bib.start.1 } else { 0 };
        let hi = if p == bib.end.0 { bib.end.1 } else { usize::MAX };
        let mut kept: Vec<Block> = Vec::new();
        for b in page.blocks.drain(..) {
            let (s, e) = (b.chars.start, b.chars.end);
            if e <= lo || s >= hi {
                kept.push(b);
                continue;
            }
            if s < lo {
                kept.push(Block { chars: s..lo, ..b.clone() });
            }
            if e > hi {
                kept.push(Block { chars: hi..e, ..b });
            }
        }
        page.blocks = kept;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse::{Page, line_blocks};

    fn doc(pages: &[&str]) -> ParsedDoc {
        ParsedDoc {
            pages: pages
                .iter()
                .enumerate()
                .map(|(i, t)| Page { pdf_page: Some(i as u32 + 1), label: None, text: t.to_string(), blocks: line_blocks(t, |_| None) })
                .collect(),
            ..ParsedDoc::default()
        }
    }

    const BODY: &str = "Introduction\nSelf-attention takes quadratic time in sequence length, which matters for long documents.\nWe study this cost carefully over many pages of discussion and results.\n";

    #[test]
    fn numbered_list_is_split_and_figures_after_it_stay_in_the_body() {
        let d = doc(&[
            &BODY.repeat(3),
            "References\n1. Parry C, Kent EE. Cancer survivors: a booming\npopulation. Cancer Epidemiol Biomarkers Prev. 2011;20(10):1996–2005.\n2. Vaswani A, et al. Attention is all you need. NeurIPS 2017. doi:10.5555/3295222.\n3. Gass B, Marrer E. Use of a case-mix approach. Ann Epidemiol. 2018;28(5):322–7.\n",
            "Figure 1: steps of the analysis\nTable 1 : Joinpoint by gender\n",
        ]);
        let mut d = d;
        let bib = detect(&d).expect("bibliography");
        assert_eq!(bib.entries.len(), 3);
        assert_eq!(bib.entries[0].number, Some(1));
        assert!(bib.entries[0].text.starts_with("Parry C, Kent EE. Cancer survivors: a booming population."), "{}", bib.entries[0].text);
        assert_eq!(bib.entries[0].year, Some(2011));
        assert_eq!(bib.entries[1].doi.as_deref(), Some("10.5555/3295222"));
        assert_eq!((bib.start, bib.end), ((1, 0), (2, 0)));
        remove_from_body(&mut d, &bib);
        assert!(d.pages[1].blocks.is_empty(), "the reference page has no body text left");
        assert!(!d.pages[2].blocks.is_empty(), "figures and tables after the references stay");
    }

    #[test]
    fn inline_numbered_entries_and_publisher_boilerplate() {
        let d = doc(&[
            &BODY.repeat(4),
            "Conflicts of Interest: none.\nReferences\n1. Yao, W.; Mei, C. Evaluation of DNA degradation. Gene 2016, 590, 142–148. [CrossRef] 2. Conroy, T.; Bosset, J.F. Neoadjuvant chemotherapy with FOLFIRINOX. Lancet Oncol. 2021, 22, 702–715. [CrossRef] [PubMed]\n3. Kim, J. Another study of ctDNA in surgery. J. Clin. Med. 2020, 9, 11.\nDisclaimer/Publisher’s Note: The statements are those of the authors.\n",
        ]);
        let bib = detect(&d).unwrap();
        assert_eq!(bib.entries.len(), 3);
        assert_eq!(bib.entries[1].text, "Conroy, T.; Bosset, J.F. Neoadjuvant chemotherapy with FOLFIRINOX. Lancet Oncol. 2021, 22, 702–715.");
        assert!(!bib.entries[2].text.contains("Disclaimer"));
    }

    #[test]
    fn a_wrapped_page_range_does_not_hide_the_next_number() {
        let d = doc(&[
            &BODY.repeat(3),
            "References\n1. First A. One study of things. J One. 2010;1:1–2.\n2. Gibson TM. Body mass index and cancer. J Clin Oncol. 2014;32(35):4004–\n11. \n3. Utada M. Incidence of multiple primary cancers. Cancer Sci. 2014;105:890–6.\n4. Last B. Final entry here in the list. J Four. 2015;4:4.\n",
        ]);
        let bib = detect(&d).unwrap();
        assert_eq!(bib.entries.iter().map(|e| e.number.unwrap()).collect::<Vec<_>>(), [1, 2, 3, 4]);
        assert!(bib.entries[1].text.ends_with("4004– 11."), "{}", bib.entries[1].text);
    }

    #[test]
    fn author_year_style() {
        let d = doc(&[
            &BODY.repeat(3),
            "# References\nSmith, J. (2019). A long title about attention\nmechanisms in practice. Journal of Things, 3, 1–10.\nJones, K. (2020). Another paper on cost. Proceedings of Stuff.\n",
        ]);
        let bib = detect(&d).unwrap();
        assert_eq!(bib.entries.len(), 2);
        assert!(bib.entries[0].text.contains("mechanisms in practice"));
        assert_eq!(bib.entries[1].year, Some(2020));
    }

    #[test]
    fn no_bibliography_or_one_mentioned_early() {
        assert!(detect(&doc(&[&BODY.repeat(4)])).is_none());
        let early = format!("References\n1. Something early that is not a bibliography at all.\n{}", BODY.repeat(6));
        assert!(detect(&doc(&[&early])).is_none(), "a References line in the first part isn't the bibliography");
    }
}
