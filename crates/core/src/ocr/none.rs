//! No OCR engine on this platform: scanned pages stay "needs OCR".

pub const ENGINE: &str = "none";
pub const NAME: &str = "OCR";

pub fn available() -> anyhow::Result<()> {
    anyhow::bail!("OCR isn't available on this platform; scanned pages stay \"needs OCR\"")
}

pub fn recognize(_png: &[u8]) -> anyhow::Result<Vec<(String, f32, [f32; 4])>> {
    available().map(|()| Vec::new())
}
