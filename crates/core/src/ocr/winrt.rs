//! Windows.Media.Ocr text recognition (Windows 10 and later).
//!
//! The engine uses the user's display languages; it needs the "Optical character recognition"
//! part of a language pack, which Windows installs with most display languages. It reports no
//! per-line confidence, so lines get 1.0.

use anyhow::Context;
use windows::Graphics::Imaging::{BitmapAlphaMode, BitmapDecoder, BitmapInterpolationMode, BitmapPixelFormat, BitmapTransform, ColorManagementMode, ExifOrientationMode, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Storage::Streams::{DataWriter, InMemoryRandomAccessStream};

pub const ENGINE: &str = "windows-media-ocr-1";
pub const NAME: &str = "Windows OCR";

const NO_LANGUAGE: &str = "Windows OCR has no recognizer for your display languages. In Settings → Time & language → Language & region, add a language pack that includes \"Optical character recognition\" (English is one), then try again";

fn engine() -> anyhow::Result<OcrEngine> {
    OcrEngine::TryCreateFromUserProfileLanguages().ok().context(NO_LANGUAGE)
}

pub fn available() -> anyhow::Result<()> {
    engine().map(|_| ())
}

pub fn recognize(png: &[u8]) -> anyhow::Result<Vec<(String, f32, [f32; 4])>> {
    let engine = engine()?;
    let bitmap = decode(png, OcrEngine::MaxImageDimension()?).context("decoding the page image")?;
    let (w, h) = (bitmap.PixelWidth()? as f32, bitmap.PixelHeight()? as f32);
    let result = engine.RecognizeAsync(&bitmap)?.get().context("Windows OCR failed")?;
    let mut out = Vec::new();
    for line in result.Lines()? {
        let text = line.Text()?.to_string_lossy();
        // A line's box is the union of its words' boxes, in pixels.
        let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for word in line.Words()? {
            let r = word.BoundingRect()?;
            (x0, y0, x1, y1) = (x0.min(r.X), y0.min(r.Y), x1.max(r.X + r.Width), y1.max(r.Y + r.Height));
        }
        if text.trim().is_empty() || x0 > x1 {
            continue;
        }
        out.push((text, 1.0, [x0 / w, y0 / h, (x1 - x0) / w, (y1 - y0) / h]));
    }
    Ok(out)
}

/// PNG → BGRA bitmap, scaled down to fit within `max` pixels on each side.
fn decode(png: &[u8], max: u32) -> windows::core::Result<SoftwareBitmap> {
    let stream = InMemoryRandomAccessStream::new()?;
    let writer = DataWriter::CreateDataWriter(&stream)?;
    writer.WriteBytes(png)?;
    writer.StoreAsync()?.get()?;
    writer.FlushAsync()?.get()?;
    writer.DetachStream()?;
    stream.Seek(0)?;
    let decoder = BitmapDecoder::CreateAsync(&stream)?.get()?;
    let (w, h) = (decoder.PixelWidth()?, decoder.PixelHeight()?);
    let transform = BitmapTransform::new()?;
    let scale = (max as f64 / w.max(h) as f64).min(1.0);
    if scale < 1.0 {
        transform.SetScaledWidth(((w as f64 * scale) as u32).max(1))?;
        transform.SetScaledHeight(((h as f64 * scale) as u32).max(1))?;
        transform.SetInterpolationMode(BitmapInterpolationMode::Fant)?;
    }
    decoder
        .GetSoftwareBitmapTransformedAsync(BitmapPixelFormat::Bgra8, BitmapAlphaMode::Premultiplied, &transform, ExifOrientationMode::IgnoreExifOrientation, ColorManagementMode::DoNotColorManage)?
        .get()
}
