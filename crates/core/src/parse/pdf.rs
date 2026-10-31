use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{Context, bail};
use pdfium_render::prelude::*;

use super::{Block, DocMeta, FileError, Page, ParsedDoc};

pub const PARSER_VERSION: i64 = 1;

/// Pages with fewer non-space characters than this, and an image on them, are scanned (§6).
const SCANNED_MAX_CHARS: usize = 25;

static PDFIUM: OnceLock<Pdfium> = OnceLock::new();

/// The process-wide PDFium binding. PDFium can only be bound once per process.
pub fn pdfium(lib: &Path) -> anyhow::Result<&'static Pdfium> {
    if let Some(p) = PDFIUM.get() {
        return Ok(p);
    }
    let bindings = Pdfium::bind_to_library(lib).with_context(|| format!("loading {}", lib.display()))?;
    Ok(PDFIUM.get_or_init(|| Pdfium::new(bindings)))
}

/// Finds libpdfium.dylib: the configured path, `$LOCALRAG_PDFIUM`, or a bundled copy
/// (see `resources::find`).
pub fn find_library(configured: Option<&Path>) -> anyhow::Result<PathBuf> {
    let explicit = configured.map(Path::to_path_buf).or_else(|| std::env::var_os("LOCALRAG_PDFIUM").map(PathBuf::from));
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p);
        }
        bail!("libpdfium not found at {}", p.display());
    }
    crate::resources::find("pdfium/lib/libpdfium.dylib")
        .context("libpdfium.dylib not found; set pdfium_lib in the global config or run scripts/fetch-pdfium.sh")
}

/// Cheap checks that don't need PDFium: empty file, or no `%PDF-` header in the first KB.
pub fn sniff(bytes: &[u8]) -> Result<(), FileError> {
    if bytes.is_empty() {
        return Err(FileError("the file is empty".into()));
    }
    if !bytes[..bytes.len().min(1024)].windows(5).any(|w| w == b"%PDF-") {
        return Err(FileError("not a PDF: the file has no PDF header (%PDF-). It may be another format renamed to .pdf".into()));
    }
    Ok(())
}

/// In-process parse. Only the sandboxed helper (`localrag-pdf`) calls this; everything else
/// goes through `crate::pdfhelper`.
pub fn parse(pdfium: &Pdfium, bytes: &[u8]) -> Result<ParsedDoc, FileError> {
    sniff(bytes)?;
    let doc = pdfium.load_pdf_from_byte_slice(bytes, None).map_err(|e| FileError(open_error(&e)))?;
    if doc.pages().is_empty() {
        return Err(FileError("the PDF has no pages".into()));
    }

    let mut parsed = ParsedDoc { meta: metadata(&doc), ..ParsedDoc::default() };
    for (i, page) in doc.pages().iter().enumerate() {
        let pdf_page = i as u32 + 1;
        let text = page
            .text()
            .map_err(|e| FileError(format!("page {pdf_page}: cannot read text: {e}")))?
            .all()
            .replace("\r\n", "\n");
        let visible = text.chars().filter(|c| !c.is_whitespace()).count();
        if visible < SCANNED_MAX_CHARS && has_image(&page) {
            parsed.scanned_pages.push(pdf_page);
        }
        let blocks = paragraph_blocks(&text);
        parsed.pages.push(Page { pdf_page: Some(pdf_page), label: page.label().map(str::to_string), text, blocks });
    }
    Ok(parsed)
}

pub fn open_error(e: &PdfiumError) -> String {
    use PdfiumInternalError as I;
    match e {
        PdfiumError::PdfiumLibraryInternalError(I::PasswordError) => {
            "the PDF is password-protected; remove the password (e.g. Preview → Export as PDF) and try again".into()
        }
        PdfiumError::PdfiumLibraryInternalError(I::SecurityError) => "the PDF's security settings don't allow reading it".into(),
        PdfiumError::PdfiumLibraryInternalError(I::FormatError) => {
            "the PDF is damaged or not a valid PDF (PDFium couldn't parse its structure); try re-saving or re-downloading it".into()
        }
        other => format!("PDFium couldn't open the PDF: {other}"),
    }
}

fn has_image(page: &PdfPage) -> bool {
    page.objects().iter().any(|o| o.object_type() == PdfPageObjectType::Image)
}

/// PDF text has no paragraph markup: blank lines separate paragraphs, otherwise each line
/// is a block and the chunker joins them into sentences.
fn paragraph_blocks(text: &str) -> Vec<Block> {
    super::line_blocks(text, |_| None)
        .into_iter()
        .map(|mut b| {
            b.lines = None;
            b
        })
        .collect()
}

fn metadata(doc: &PdfDocument) -> DocMeta {
    let tag = |t| doc.metadata().get(t).map(|m| m.value().trim().to_string()).filter(|v| !v.is_empty());
    let mut meta = DocMeta {
        title: tag(PdfDocumentMetadataTagType::Title),
        authors: tag(PdfDocumentMetadataTagType::Author)
            .map(|a| a.split([';', ',']).map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect())
            .unwrap_or_default(),
        // No year: the PDF creation date is when the file was made, not when the work was
        // published. Crossref supplies it (§7).
        ..DocMeta::default()
    };
    if meta.title.is_some() || !meta.authors.is_empty() {
        meta.source = Some("pdf".into());
    }
    meta
}
