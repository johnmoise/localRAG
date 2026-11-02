use super::{FileError, Page, ParsedDoc, line_blocks};

pub const PARSER_VERSION: i64 = 1;

/// Paragraphs separated by blank lines; citations use line ranges.
pub fn parse(bytes: &[u8]) -> Result<ParsedDoc, FileError> {
    let text = super::utf8(bytes)?;
    let blocks = line_blocks(&text, |_| None);
    Ok(ParsedDoc { pages: vec![Page { text, blocks, ..Page::default() }], ..ParsedDoc::default() })
}
