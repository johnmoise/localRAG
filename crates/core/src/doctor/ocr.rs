use anyhow::ensure;

use super::Outcome;

/// Runs the platform's OCR engine on a rendered page (§6).
pub(super) fn check(png: &[u8]) -> anyhow::Result<Outcome> {
    if let Err(e) = crate::ocr::available() {
        return Ok(Outcome::Warn(format!("{e:#}")));
    }
    let lines = crate::ocr::recognize(png)?;
    let text = lines.iter().map(|l| l.0.as_str()).collect::<Vec<_>>().join(" ");
    ensure!(text.to_lowercase().contains("quadratic time"), "{} text did not match the page: {text:?}", crate::ocr::NAME);
    let min_conf = lines.iter().map(|l| l.1).fold(1.0f32, f32::min);
    Ok(Outcome::Pass(format!("{} lines recognized, lowest confidence {min_conf:.2}", lines.len())))
}
