//! Parsing `[S3 "quote"]` markers and verifying quotes against their sources
//! (ARCHITECTURE.md §10).

use std::ops::Range;

use unicode_segmentation::UnicodeSegmentation;

/// One bracketed marker in an answer, e.g. `[S1 "a quote"; S2]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Marker {
    /// Byte range of the whole bracket in the answer.
    pub range: Range<usize>,
    pub items: Vec<MarkerItem>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MarkerItem {
    /// 1-based source number (`S3` → 3).
    pub source: usize,
    pub quote: Option<String>,
}

pub fn parse_markers(text: &str) -> Vec<Marker> {
    let mut out = Vec::new();
    let mut i = 0;
    while let Some(off) = text[i..].find('[') {
        let start = i + off;
        match parse_bracket(&text[start..]) {
            Some((mut len, mut items)) => {
                // Also accept the quote right after the bracket: `[S1] "quote"`.
                if items.len() == 1 && items[0].quote.is_none()
                    && let Some((qlen, quote)) = trailing_quote(&text[start + len..])
                {
                    items[0].quote = Some(quote);
                    len += qlen;
                }
                out.push(Marker { range: start..start + len, items });
                i = start + len;
            }
            None => i = start + 1,
        }
    }
    out
}

/// Parses `[S1 "q"; S2, S3]` at the start of `s`; returns (byte length, items).
fn parse_bracket(s: &str) -> Option<(usize, Vec<MarkerItem>)> {
    let mut chars = s.char_indices().peekable();
    chars.next(); // '['
    let mut items = Vec::new();
    loop {
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        let (_, c) = chars.next()?;
        if c != 'S' && c != 's' {
            return None;
        }
        let mut digits = String::new();
        while let Some((_, d)) = chars.next_if(|(_, c)| c.is_ascii_digit()) {
            digits.push(d);
        }
        let source: usize = digits.parse().ok().filter(|&n| n > 0)?;
        while chars.next_if(|(_, c)| c.is_whitespace() || *c == ':').is_some() {}
        let mut quote = None;
        if let Some(close) = chars.peek().and_then(|&(_, q)| closing_quote(q)) {
            chars.next();
            let mut body = String::new();
            loop {
                let (_, c) = chars.next()?;
                if c == close || (close == '"' && c == '”') {
                    break;
                }
                body.push(c);
            }
            quote = Some(body.trim().to_string()).filter(|q| !q.is_empty());
        }
        items.push(MarkerItem { source, quote });
        while chars.next_if(|(_, c)| c.is_whitespace()).is_some() {}
        match chars.next()? {
            (idx, ']') => return Some((idx + 1, items)),
            (_, ',' | ';') => continue,
            _ => return None,
        }
    }
}

/// A quoted string (after optional spaces or a colon) at the start of `s`: (byte length, text).
fn trailing_quote(s: &str) -> Option<(usize, String)> {
    let lead = s.len() - s.trim_start_matches([' ', ':']).len();
    let rest = &s[lead..];
    let open = rest.chars().next()?;
    let close = closing_quote(open).filter(|_| open != '\'')?;
    let body_start = open.len_utf8();
    let end = rest[body_start..].find(|c| c == close || (close == '"' && c == '”'))?;
    let quote = rest[body_start..body_start + end].trim();
    let closing_len = rest[body_start + end..].chars().next()?.len_utf8();
    (!quote.is_empty() && quote.split_whitespace().count() <= 40)
        .then(|| (lead + body_start + end + closing_len, quote.to_string()))
}

fn closing_quote(open: char) -> Option<char> {
    match open {
        '"' => Some('"'),
        '“' => Some('”'),
        '\'' => Some('\''),
        '‘' => Some('’'),
        _ => None,
    }
}

/// The answer with each marker item replaced by `⟦k⟧`, k being the item's ordinal in
/// reading order (`Citation::ordinal`). The UI renders the Markdown, then swaps the
/// placeholders for citation chips.
pub fn placeholders(text: &str) -> String {
    let mut out = String::new();
    let mut i = 0;
    let mut ordinal = 0;
    for m in parse_markers(text) {
        out.push_str(&text[i..m.range.start]);
        for _ in &m.items {
            out.push_str(&format!("⟦{ordinal}⟧"));
            ordinal += 1;
        }
        i = m.range.end;
    }
    out.push_str(&text[i..]);
    out
}

/// Where a quote sits in a chunk: a char range in the chunk text.
pub fn verify_quote(quote: &str, chunk: &str, ocr: bool) -> Option<Range<usize>> {
    let parts: Vec<&str> = quote.split(['…']).flat_map(|p| p.split("...")).map(str::trim).filter(|p| !p.is_empty()).collect();
    if parts.is_empty() {
        return None;
    }
    let words = words(chunk);
    let mut from = 0;
    let mut span: Option<Range<usize>> = None;
    for part in parts {
        let want: Vec<String> = words_of(part);
        if want.is_empty() {
            continue;
        }
        let found = find_words(&words[from..], &want).or_else(|| ocr.then(|| find_words_fuzzy(&words[from..], &want)).flatten())?;
        let (a, b) = (from + found.start, from + found.end);
        let r = words[a].1.start..words[b - 1].1.end;
        span = Some(match span {
            None => r,
            Some(s) => s.start..r.end,
        });
        from = b;
    }
    span
}

/// The chunk sentence sharing the most content words with `claim`, when a citation has no
/// usable quote.
pub fn best_sentence(claim: &str, chunk: &str) -> Option<Range<usize>> {
    let claim_words: Vec<String> = words_of(claim).into_iter().filter(|w| w.chars().count() > 3).collect();
    if claim_words.is_empty() {
        return None;
    }
    let mut best: Option<(usize, Range<usize>)> = None;
    for (byte_off, sentence) in chunk.split_sentence_bound_indices() {
        let s_words = words_of(sentence);
        let score = claim_words.iter().filter(|w| s_words.contains(w)).count();
        if score > 0 && best.as_ref().is_none_or(|(b, _)| score > *b) {
            let lead = sentence.chars().take_while(|c| c.is_whitespace()).count();
            let start = chunk[..byte_off].chars().count() + lead;
            best = Some((score, start..start + sentence.trim().chars().count()));
        }
    }
    best.map(|(_, r)| r)
}

/// Lowercased alphanumeric words with their char ranges. Ligatures are expanded, so a quote
/// typed with "fi" matches a PDF's "ﬁ".
fn words(text: &str) -> Vec<(String, Range<usize>)> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut start = 0;
    for (i, c) in text.chars().enumerate() {
        let expanded = match c {
            'ﬁ' => Some("fi"),
            'ﬂ' => Some("fl"),
            'ﬀ' => Some("ff"),
            'ﬃ' => Some("ffi"),
            'ﬄ' => Some("ffl"),
            _ => None,
        };
        if c.is_alphanumeric() || expanded.is_some() {
            if current.is_empty() {
                start = i;
            }
            match expanded {
                Some(e) => current.push_str(e),
                None => current.extend(c.to_lowercase()),
            }
        } else if !current.is_empty() {
            out.push((std::mem::take(&mut current), start..i));
        }
    }
    if !current.is_empty() {
        out.push((current, start..text.chars().count()));
    }
    out
}

fn words_of(text: &str) -> Vec<String> {
    words(text).into_iter().map(|w| w.0).collect()
}

fn find_words(hay: &[(String, Range<usize>)], want: &[String]) -> Option<Range<usize>> {
    if want.len() > hay.len() {
        return None;
    }
    let last = want.len() - 1;
    // Models sometimes cut a quote mid-word ("commonly use" for "commonly used"): the final
    // word may be a prefix of at least 3 characters.
    let matches = |i: usize, j: usize, w: &String| {
        let h = &hay[i + j].0;
        h == w || (j == last && j > 0 && w.chars().count() >= 3 && h.starts_with(w.as_str()))
    };
    (0..=hay.len() - want.len())
        .find(|&i| want.iter().enumerate().all(|(j, w)| matches(i, j, w)))
        .map(|i| i..i + want.len())
}

/// OCR text: accept a window where at least 80% of the words match (§6).
fn find_words_fuzzy(hay: &[(String, Range<usize>)], want: &[String]) -> Option<Range<usize>> {
    if want.len() < 3 || want.len() > hay.len() {
        return None;
    }
    let need = (want.len() as f32 * 0.8).ceil() as usize;
    (0..=hay.len() - want.len())
        .map(|i| (i, hay[i..i + want.len()].iter().zip(want).filter(|(h, w)| &h.0 == *w).count()))
        .filter(|&(_, n)| n >= need)
        .max_by_key(|&(_, n)| n)
        .map(|(i, _)| i..i + want.len())
}

/// Factual sentences of an answer that carry no citation marker. Headings, list
/// lead-ins ending with ':', and very short fragments are not counted.
pub fn uncited_sentences(text: &str, markers: &[Marker]) -> Vec<String> {
    let mut out = Vec::new();
    let mut sentences: Vec<(Range<usize>, &str)> = text
        .split_sentence_bound_indices()
        .map(|(i, s)| (i..i + s.len(), s))
        .collect();
    // A marker after the full stop ("Fact. [S1]") belongs to the sentence before it.
    let mut cited = vec![false; sentences.len()];
    for m in markers {
        if let Some(idx) = sentences.iter().position(|(r, _)| r.contains(&m.range.start)) {
            let before = &text[sentences[idx].0.start..m.range.start];
            let target = if before.trim().is_empty() && idx > 0 { idx - 1 } else { idx };
            cited[target] = true;
        }
    }
    for (i, (range, _)) in sentences.drain(..).enumerate() {
        let stripped = strip_markers(&text[range.clone()], markers, range.start);
        let t = stripped.trim();
        let letters = t.chars().filter(|c| c.is_alphabetic()).count();
        if cited[i] || letters < 20 || t.starts_with('#') || t.ends_with(':') || t.starts_with("INSUFFICIENT") {
            continue;
        }
        out.push(t.trim_start_matches(['-', '*', ' ']).to_string());
    }
    out
}

fn strip_markers(s: &str, markers: &[Marker], offset: usize) -> String {
    let mut out = String::new();
    let mut i = 0;
    for m in markers.iter().filter(|m| m.range.start >= offset && m.range.end <= offset + s.len()) {
        out.push_str(&s[i..m.range.start - offset]);
        i = m.range.end - offset;
    }
    out.push_str(&s[i..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_marker_forms() {
        let t = r#"A fact [S1 "exact words"]. Two [S2; S3 “curly quote”]. Bad [X1] and [S0] and [see S1]. Bare [S4]"#;
        let m = parse_markers(t);
        assert_eq!(m.len(), 3);
        assert_eq!(m[0].items, [MarkerItem { source: 1, quote: Some("exact words".into()) }]);
        assert_eq!(m[1].items, [MarkerItem { source: 2, quote: None }, MarkerItem { source: 3, quote: Some("curly quote".into()) }]);
        assert_eq!(m[2].items, [MarkerItem { source: 4, quote: None }]);
        assert_eq!(&t[m[0].range.clone()], r#"[S1 "exact words"]"#);

        let after = r#"Fact [S2] "quoted after the bracket" and more."#;
        let m = parse_markers(after);
        assert_eq!(m[0].items, [MarkerItem { source: 2, quote: Some("quoted after the bracket".into()) }]);
        assert_eq!(&after[m[0].range.clone()], r#"[S2] "quoted after the bracket""#);
    }

    #[test]
    fn placeholders_number_items_in_order() {
        assert_eq!(placeholders(r#"A [S1 "q"]. B [S2; S3] and [S1] "after"."#), "A ⟦0⟧. B ⟦1⟧⟦2⟧ and ⟦3⟧.");
    }

    #[test]
    fn verifies_quotes_ignoring_case_punctuation_and_ligatures() {
        let chunk = "The SIR can be interpreted as the “relative risk” of SPC; it is signiﬁcant.";
        let r = verify_quote("interpreted as the relative risk", chunk, false).unwrap();
        let span: String = chunk.chars().skip(r.start).take(r.len()).collect();
        assert_eq!(span, "interpreted as the “relative risk");
        assert!(verify_quote("it is significant", chunk, false).is_some());
        assert!(verify_quote("it is signif", chunk, false).is_some(), "truncated final word");
        assert!(verify_quote("is signif it", chunk, false).is_none(), "only the final word may be cut");
        assert!(verify_quote("The SIR ... significant", chunk, false).is_some());
        assert!(verify_quote("the absolute risk", chunk, false).is_none());
        assert!(verify_quote("can be interpretcd as the relatlve risk", chunk, false).is_none());
        assert!(verify_quote("can be interpretcd as the relative risk", chunk, true).is_some(), "OCR tolerance");
    }

    #[test]
    fn finds_uncited_sentences() {
        let t = "Synchronous cancers are common. [S1 \"x\"] They are defined within six months [S2]. This sentence has no citation at all.\n\n## Summary:\nShort.";
        let m = parse_markers(t);
        assert_eq!(uncited_sentences(t, &m), ["This sentence has no citation at all."]);
    }

    #[test]
    fn best_sentence_by_overlap() {
        let chunk = "Cancer survivors are growing in number. Survival depends on stage at diagnosis.";
        let r = best_sentence("Survival depends strongly on the stage", chunk).unwrap();
        assert_eq!(chunk.chars().skip(r.start).take(r.len()).collect::<String>(), "Survival depends on stage at diagnosis.");
    }
}
