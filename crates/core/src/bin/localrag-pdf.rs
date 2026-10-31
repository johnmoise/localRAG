//! `localrag-pdf`: the PDF helper process (see `localrag_core::pdfhelper`). Reads one JSON
//! request line from stdin, writes one JSON reply line to stdout, and exits.

use std::io::{BufRead, Write};
use std::path::PathBuf;

use localrag_core::parse::{FileError, pdf};
use localrag_core::pdfhelper::{PageChars, encode_hex};
use pdfium_render::prelude::*;
use serde_json::{Value, json};

fn main() {
    let reply = match run() {
        Ok(v) => v,
        Err(Failure::File(e)) => json!({ "file_error": e.0 }),
        Err(Failure::Other(e)) => json!({ "error": format!("{e:#}") }),
    };
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "{reply}");
}

enum Failure {
    File(FileError),
    Other(anyhow::Error),
}

impl From<anyhow::Error> for Failure {
    fn from(e: anyhow::Error) -> Self {
        Failure::Other(e)
    }
}

impl From<FileError> for Failure {
    fn from(e: FileError) -> Self {
        Failure::File(e)
    }
}

fn run() -> Result<Value, Failure> {
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line).map_err(anyhow::Error::from)?;
    let req: Value = serde_json::from_str(&line).map_err(anyhow::Error::from)?;
    let lib = pdf::find_library(None)?;
    let pdfium = pdf::pdfium(&lib)?;
    let path = PathBuf::from(req["path"].as_str().unwrap_or_default());
    let bytes = std::fs::read(&path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))?;

    match req["op"].as_str() {
        Some("parse") => Ok(json!({ "doc": pdf::parse(pdfium, &bytes)? })),
        Some("chars") => {
            let doc = open(pdfium, &bytes)?;
            let (first, last) = (req["first"].as_u64().unwrap_or(1) as u32, req["last"].as_u64().unwrap_or(1) as u32);
            let mut pages = Vec::new();
            for p in first..=last.max(first) {
                let page = doc.pages().get((p - 1) as i32).map_err(|e| FileError(format!("page {p}: {e}")))?;
                let text = page.text().map_err(|e| FileError(format!("page {p}: {e}")))?;
                let chars = text
                    .chars()
                    .iter()
                    .map(|c| {
                        let b = c.loose_bounds().ok().map(|r| [r.left().value, r.bottom().value, r.right().value, r.top().value]);
                        (c.unicode_char().unwrap_or(' '), b)
                    })
                    .collect();
                pages.push(PageChars { pdf_page: p, width: page.width().value, height: page.height().value, chars });
            }
            Ok(json!({ "pages": pages }))
        }
        Some("render") => {
            let doc = open(pdfium, &bytes)?;
            let p = req["page"].as_u64().unwrap_or(1) as i32;
            let width = req["width"].as_u64().unwrap_or(2550) as i32;
            let page = doc.pages().get(p - 1).map_err(|e| FileError(format!("page {p}: {e}")))?;
            let image = page
                .render_with_config(&PdfRenderConfig::new().set_target_width(width))
                .and_then(|b| b.as_image())
                .map_err(|e| FileError(format!("page {p} couldn't be rendered: {e}")))?;
            let mut png = Vec::new();
            image.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(anyhow::Error::from)?;
            Ok(json!({ "png_hex": encode_hex(&png), "width": image.width(), "height": image.height() }))
        }
        other => Err(anyhow::anyhow!("unknown op {other:?}").into()),
    }
}

fn open<'a>(pdfium: &'a Pdfium, bytes: &'a [u8]) -> Result<PdfDocument<'a>, FileError> {
    pdf::sniff(bytes)?;
    pdfium.load_pdf_from_byte_slice(bytes, None).map_err(|e| FileError(pdf::open_error(&e)))
}
