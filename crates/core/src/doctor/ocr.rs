use anyhow::{Context, ensure};
use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSData, NSDictionary};
use objc2_vision::{
    VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel,
};

use super::Outcome;

/// Runs Apple Vision text recognition on a rendered page (§6).
pub(super) fn check(png: &[u8]) -> anyhow::Result<Outcome> {
    let lines = recognize(png)?;
    let text = lines.iter().map(|l| l.0.as_str()).collect::<Vec<_>>().join(" ");
    ensure!(
        text.to_lowercase().contains("quadratic time"),
        "Vision text did not match the page: {text:?}"
    );
    let min_conf = lines.iter().map(|l| l.1).fold(1.0f32, f32::min);
    Ok(Outcome::Pass(format!("{} lines recognized, lowest confidence {min_conf:.2}", lines.len())))
}

/// (text, confidence) for each recognized line.
fn recognize(png: &[u8]) -> anyhow::Result<Vec<(String, f32)>> {
    let data = NSData::with_bytes(png);
    let handler = VNImageRequestHandler::initWithData_options(
        VNImageRequestHandler::alloc(),
        &data,
        &NSDictionary::new(),
    );
    let request = VNRecognizeTextRequest::new();
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
    request.setUsesLanguageCorrection(true);
    let as_request: Retained<VNRequest> = Retained::into_super(Retained::into_super(request.clone()));
    handler
        .performRequests_error(&NSArray::from_retained_slice(&[as_request]))
        .map_err(|e| anyhow::anyhow!("Vision request failed: {e}"))?;

    let results = request.results().context("Vision returned no results")?;
    let mut lines = Vec::new();
    for obs in results.iter() {
        if let Some(best) = obs.topCandidates(1).firstObject() {
            lines.push((best.string().to_string(), best.confidence()));
        }
    }
    Ok(lines)
}
