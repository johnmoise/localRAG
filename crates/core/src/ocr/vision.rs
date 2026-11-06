//! Apple Vision text recognition (macOS).

use objc2::AnyThread;
use objc2::rc::Retained;
use objc2_foundation::{NSArray, NSData, NSDictionary};
use objc2_vision::{VNImageRequestHandler, VNRecognizeTextRequest, VNRequest, VNRequestTextRecognitionLevel};

pub const ENGINE: &str = "apple-vision-accurate-1";
pub const NAME: &str = "Apple Vision";

pub fn available() -> anyhow::Result<()> {
    Ok(())
}

pub fn recognize(png: &[u8]) -> anyhow::Result<Vec<(String, f32, [f32; 4])>> {
    let data = NSData::with_bytes(png);
    let handler = VNImageRequestHandler::initWithData_options(VNImageRequestHandler::alloc(), &data, &NSDictionary::new());
    let request = VNRecognizeTextRequest::new();
    request.setRecognitionLevel(VNRequestTextRecognitionLevel::Accurate);
    request.setUsesLanguageCorrection(true);
    let as_request: Retained<VNRequest> = Retained::into_super(Retained::into_super(request.clone()));
    handler
        .performRequests_error(&NSArray::from_retained_slice(&[as_request]))
        .map_err(|e| anyhow::anyhow!("Vision request failed: {e}"))?;
    let Some(results) = request.results() else { return Ok(Vec::new()) };
    let mut out = Vec::new();
    for obs in results.iter() {
        let Some(best) = obs.topCandidates(1).firstObject() else { continue };
        // Vision's box is normalized with the origin at the bottom-left.
        let r = unsafe { obs.boundingBox() };
        let (x, y, w, h) = (r.origin.x as f32, r.origin.y as f32, r.size.width as f32, r.size.height as f32);
        out.push((best.string().to_string(), best.confidence(), [x, 1.0 - y - h, w, h]));
    }
    Ok(out)
}
