//! `localrag doctor`: checks that every external piece the plan depends on works on this
//! machine (ARCHITECTURE.md §18, Phase 0).

mod crossref;
mod ocr;
mod ollama;
mod pdf;
mod sqlite;
mod tokens;

use std::path::PathBuf;
use std::time::Instant;

/// Small text PDF with page labels starting at 345 (see scripts/make-fixture-pdf.py).
pub const SAMPLE_PDF: &[u8] = include_bytes!("../../../../tests/fixtures/sample.pdf");

#[derive(Debug, Clone)]
pub struct DoctorConfig {
    pub ollama_host: String,
    pub embed_model: String,
    pub chat_model: String,
    pub num_ctx: u32,
    pub pdfium_lib: PathBuf,
    pub tokenizer_file: PathBuf,
    pub crossref_email: Option<String>,
    pub skip_network: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Pass,
    Warn,
    Fail,
    Skip,
}

#[derive(Debug, Clone)]
pub struct Check {
    pub name: &'static str,
    pub status: Status,
    pub detail: String,
    pub millis: u128,
}

impl Check {
    fn from_result(name: &'static str, started: Instant, result: anyhow::Result<Outcome>) -> Self {
        let millis = started.elapsed().as_millis();
        let (status, detail) = match result {
            Ok(Outcome::Pass(d)) => (Status::Pass, d),
            Ok(Outcome::Warn(d)) => (Status::Warn, d),
            Ok(Outcome::Skip(d)) => (Status::Skip, d),
            Err(e) => (Status::Fail, format!("{e:#}")),
        };
        Check { name, status, detail, millis }
    }
}

/// What a check reports when it didn't error.
pub(crate) enum Outcome {
    Pass(String),
    Warn(String),
    Skip(String),
}

pub async fn run(cfg: &DoctorConfig) -> Vec<Check> {
    let mut checks = Vec::new();

    let t = Instant::now();
    checks.push(Check::from_result("SQLite + FTS5 + sqlite-vec", t, sqlite::check()));

    // PDFium renders page 1 for the OCR check, so OCR depends on it.
    let t = Instant::now();
    let pdf = pdf::check(&cfg.pdfium_lib);
    let rendered = pdf.as_ref().ok().map(|r| r.page1_png.clone());
    checks.push(Check::from_result("PDFium text, boxes, labels, render", t, pdf.map(|r| r.outcome)));

    let t = Instant::now();
    let ocr = match rendered {
        Some(png) => ocr::check(&png),
        None => Ok(Outcome::Skip("needs PDFium to render a page".into())),
    };
    checks.push(Check::from_result("Apple Vision OCR", t, ocr));

    let client = reqwest::Client::new();
    let t = Instant::now();
    let reachable = ollama::check_reachable(&client, &cfg.ollama_host).await;
    let ollama_up = reachable.is_ok();
    checks.push(Check::from_result("Ollama reachable", t, reachable));

    if ollama_up {
        let t = Instant::now();
        checks.push(Check::from_result("Embedding model", t, ollama::check_embed(&client, cfg).await));
        let t = Instant::now();
        checks.push(Check::from_result("Chat with explicit num_ctx", t, ollama::check_chat(&client, cfg).await));
        let t = Instant::now();
        checks.push(Check::from_result("Token counts vs Ollama", t, tokens::check(&client, cfg).await));
    } else {
        for name in ["Embedding model", "Chat with explicit num_ctx", "Token counts vs Ollama"] {
            checks.push(Check { name, status: Status::Skip, detail: "Ollama not reachable".into(), millis: 0 });
        }
    }

    let t = Instant::now();
    let crossref = if cfg.skip_network {
        Ok(Outcome::Skip("--offline".into()))
    } else {
        crossref::check(&client, cfg.crossref_email.as_deref()).await
    };
    checks.push(Check::from_result("Crossref lookup", t, crossref));

    checks
}
