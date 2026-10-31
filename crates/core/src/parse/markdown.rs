use super::{DocMeta, FileError, Page, ParsedDoc, line_blocks};

pub const PARSER_VERSION: i64 = 1;

/// ATX headings (`#`…`######`) define sections; fenced code is never read as a heading.
/// YAML front matter supplies title, author(s), year and doi.
pub fn parse(bytes: &[u8]) -> Result<ParsedDoc, FileError> {
    let text = super::utf8(bytes)?;
    let (meta, body_start) = front_matter(&text);
    let mut blocks = line_blocks(&text, heading);
    // Drop blocks that are the front matter itself.
    let body_start_chars = text[..body_start].chars().count();
    blocks.retain(|b| b.chars.start >= body_start_chars);
    Ok(ParsedDoc { pages: vec![Page { text, blocks, ..Page::default() }], meta, ..ParsedDoc::default() })
}

fn heading(line: &str) -> Option<(usize, String)> {
    let level = line.chars().take_while(|&c| c == '#').count();
    if !(1..=6).contains(&level) {
        return None;
    }
    let rest = &line[level..];
    if !rest.is_empty() && !rest.starts_with(' ') {
        return None;
    }
    let title = rest.trim().trim_end_matches('#').trim();
    Some((level, title.to_string()))
}

/// Minimal front matter reader: `key: value` lines and `- item` lists between `---` fences.
/// Returns the metadata and the byte offset where the body starts.
fn front_matter(text: &str) -> (DocMeta, usize) {
    let mut meta = DocMeta::default();
    let Some(rest) = text.strip_prefix("---\n").or_else(|| text.strip_prefix("---\r\n")) else {
        return (meta, 0);
    };
    let Some(end) = rest.find("\n---") else { return (meta, 0) };
    let header = &rest[..end];
    let after = &rest[end + 4..];
    let body_start = text.len() - after.len() + after.find('\n').map_or(after.len(), |i| i + 1);

    let mut list_key: Option<String> = None;
    for line in header.lines() {
        if let Some(item) = line.trim_start().strip_prefix("- ") {
            if matches!(list_key.as_deref(), Some("author" | "authors")) {
                meta.authors.push(unquote(item));
            }
            continue;
        }
        let Some((key, value)) = line.split_once(':') else { continue };
        let key = key.trim().to_ascii_lowercase();
        let value = unquote(value);
        list_key = value.is_empty().then(|| key.clone());
        match key.as_str() {
            "title" if !value.is_empty() => meta.title = Some(value),
            "author" | "authors" if !value.is_empty() => {
                let inner = value.trim_start_matches('[').trim_end_matches(']');
                meta.authors.extend(inner.split(';').flat_map(|a| a.split(" and ")).map(unquote).filter(|a| !a.is_empty()));
            }
            "year" | "date" => meta.year = value.get(..4).and_then(|y| y.parse().ok()),
            "doi" if !value.is_empty() => meta.doi = Some(value),
            _ => {}
        }
    }
    if meta.title.is_some() || !meta.authors.is_empty() || meta.year.is_some() || meta.doi.is_some() {
        meta.source = Some("frontmatter".into());
    }
    (meta, body_start)
}

fn unquote(s: &str) -> String {
    s.trim().trim_matches('"').trim_matches('\'').trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sections_front_matter_and_fences() {
        let md = "---\ntitle: \"Notes on Attention\"\nauthor:\n  - A. Tester\n  - B. Writer\nyear: 2026\ndoi: 10.1/xyz\n---\n# Intro\nFirst para.\n\n## Detail\nSecond\npara.\n```\n# not a heading\n```\n";
        let doc = parse(md.as_bytes()).unwrap();
        assert_eq!(doc.meta.title.as_deref(), Some("Notes on Attention"));
        assert_eq!(doc.meta.authors, ["A. Tester", "B. Writer"]);
        assert_eq!(doc.meta.year, Some(2026));
        assert_eq!(doc.meta.doi.as_deref(), Some("10.1/xyz"));
        let page = &doc.pages[0];
        let texts: Vec<String> = page.blocks.iter().map(|b| page.text.chars().skip(b.chars.start).take(b.chars.len()).collect()).collect();
        assert_eq!(texts[0], "First para.");
        assert_eq!(page.blocks[0].section, ["Intro"]);
        assert_eq!(texts[1], "Second\npara.\n```\n# not a heading\n```");
        assert_eq!(page.blocks[1].section, ["Intro", "Detail"]);
        assert_eq!(page.blocks[1].lines, Some(13..18));
    }
}
