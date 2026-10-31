use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, ensure};
use pdfium_render::prelude::*;

use super::{Outcome, SAMPLE_PDF};

pub(super) struct PdfReport {
    pub outcome: Outcome,
    pub page1_png: Vec<u8>,
}

/// Text, per-character boxes and page labels from the fixture, plus a page render (§5, §6).
pub(super) fn check(lib: &Path) -> anyhow::Result<PdfReport> {
    let pdfium = crate::parse::pdf::pdfium(lib).context("binding libpdfium")?;
    let doc = pdfium.load_pdf_from_byte_slice(SAMPLE_PDF, None)?;

    let mut labels = Vec::new();
    let mut boxed_chars = 0usize;
    let mut page1_text = String::new();
    for (i, page) in doc.pages().iter().enumerate() {
        labels.push(page.label().unwrap_or("").to_string());
        let text = page.text()?;
        if i == 0 {
            page1_text = text.all();
        }
        boxed_chars += text.chars().iter().filter(|c| c.loose_bounds().is_ok()).count();
    }
    ensure!(page1_text.contains("quadratic time"), "unexpected page text: {page1_text:?}");
    ensure!(labels == ["345", "346"], "page labels {labels:?}, expected [\"345\", \"346\"]");
    ensure!(boxed_chars > 100, "only {boxed_chars} characters had bounding boxes");

    // ~300 DPI for a US-letter page, as planned for OCR.
    let page1 = doc.pages().get(0)?;
    let image = page1
        .render_with_config(&PdfRenderConfig::new().set_target_width(2550))?
        .as_image()?;
    let mut page1_png = Vec::new();
    image.write_to(&mut Cursor::new(&mut page1_png), image::ImageFormat::Png)?;

    Ok(PdfReport {
        outcome: Outcome::Pass(format!(
            "{} pages, {boxed_chars} chars with boxes, labels {labels:?}, page 1 rendered {}×{}",
            doc.pages().len(),
            image.width(),
            image.height()
        )),
        page1_png,
    })
}
