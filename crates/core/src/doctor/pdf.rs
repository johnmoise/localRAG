use anyhow::{Context, bail, ensure};

use super::{Outcome, SAMPLE_PDF};
use crate::pdfhelper::PdfHelper;

pub(super) struct PdfReport {
    pub outcome: Outcome,
    pub page1_png: Vec<u8>,
}

/// Through the sandboxed helper process, as the app uses it: text, per-character boxes and
/// page labels from the fixture, plus a page render (§5, §6, §14).
pub(super) fn check(helper: &PdfHelper) -> anyhow::Result<PdfReport> {
    let path = std::env::temp_dir().join(format!("localrag-doctor-{}.pdf", std::process::id()));
    std::fs::write(&path, SAMPLE_PDF)?;
    let result = (|| {
        let doc = helper.parse(&path)?.map_err(|e| anyhow::anyhow!("parse: {e}"))?;
        let labels: Vec<String> = doc.pages.iter().map(|p| p.label.clone().unwrap_or_default()).collect();
        ensure!(doc.pages[0].text.contains("quadratic time"), "unexpected page text: {:?}", doc.pages[0].text);
        ensure!(labels == ["345", "346"], "page labels {labels:?}, expected [\"345\", \"346\"]");

        let pages = helper.chars(&path, 1, 2)?.map_err(|e| anyhow::anyhow!("chars: {e}"))?;
        let boxed = pages.iter().flat_map(|p| &p.chars).filter(|c| c.1.is_some()).count();
        ensure!(boxed > 100, "only {boxed} characters had bounding boxes");

        // ~300 DPI for a US-letter page, as planned for OCR.
        let png = helper.render(&path, 1, 2550)?.map_err(|e| anyhow::anyhow!("render: {e}"))?;
        if !png.starts_with(b"\x89PNG") {
            bail!("render didn't return a PNG");
        }
        Ok(PdfReport {
            outcome: Outcome::Pass(format!(
                "{} pages, {boxed} chars with boxes, labels {labels:?}, page 1 rendered ({} KB); helper {}",
                doc.pages.len(),
                png.len() / 1024,
                if helper.is_sandboxed() { "sandboxed (no network, no writes)" } else { "NOT sandboxed" }
            )),
            page1_png: png,
        })
    })();
    let _ = std::fs::remove_file(&path);
    result.context("PDF helper")
}
