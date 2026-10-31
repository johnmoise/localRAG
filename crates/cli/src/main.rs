use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};
use localrag_core::doctor::{self, DoctorConfig};
use localrag_core::embed::OllamaEmbedder;
use localrag_core::index::{self, Change, Parsers};
use localrag_core::workspace::{self, Workspace};
use localrag_core::db;

#[derive(Parser)]
#[command(name = "localrag", version, about = "Grounded, cited Q&A over your own literature")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Create a workspace here: .localrag/ (state) and corpus/ (your documents)
    Init {
        /// Directory to initialize (default: current directory)
        dir: Option<PathBuf>,
    },
    /// Index new and changed files in corpus/; unchanged files are skipped
    Index,
    /// Show index counts, or every document with --docs
    Status {
        #[arg(long)]
        docs: bool,
    },
    /// Permanently delete a removed document's record, chunks, vectors and OCR cache
    Forget {
        /// Document id or path relative to corpus/; omit with --all-removed
        doc: Option<String>,
        /// Forget every removed document
        #[arg(long, conflicts_with = "doc")]
        all_removed: bool,
        /// Don't ask for confirmation
        #[arg(long, short)]
        yes: bool,
    },
    /// Check that PDFium, Vision, SQLite, Ollama and Crossref all work on this machine
    Doctor {
        #[arg(long, default_value = "http://127.0.0.1:11434")]
        ollama_host: String,
        #[arg(long, default_value = "qwen3-embedding:0.6b")]
        embed_model: String,
        #[arg(long, default_value = "qwen3.5:4b")]
        chat_model: String,
        #[arg(long, default_value_t = 16384)]
        num_ctx: u32,
        /// Path to libpdfium.dylib (default: discovered automatically)
        #[arg(long)]
        pdfium: Option<PathBuf>,
        /// Hugging Face tokenizer.json matching --chat-model
        #[arg(long, default_value = "third_party/tokenizers/qwen35.json")]
        tokenizer: PathBuf,
        /// Email for Crossref's polite pool; omit to use the public pool
        #[arg(long)]
        crossref_email: Option<String>,
        /// Skip checks that need the internet (Crossref)
        #[arg(long)]
        offline: bool,
    },
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Init { dir } => init(dir),
        Command::Index => index_cmd(),
        Command::Status { docs } => status(docs),
        Command::Forget { doc, all_removed, yes } => forget(doc, all_removed, yes),
        Command::Doctor { ollama_host, embed_model, chat_model, num_ctx, pdfium, tokenizer, crossref_email, offline } => {
            let cfg = DoctorConfig {
                ollama_host,
                embed_model,
                chat_model,
                num_ctx,
                pdfium_lib: pdfium,
                tokenizer_file: tokenizer,
                crossref_email,
                skip_network: offline,
            };
            return run_doctor(&cfg);
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn init(dir: Option<PathBuf>) -> anyhow::Result<()> {
    let dir = dir.unwrap_or(std::env::current_dir()?);
    let report = Workspace::init(&dir)?;
    if report.already_existed {
        println!("Workspace already exists at {}", report.root.display());
    } else {
        println!("Created workspace at {}", report.root.display());
        println!("  Put .pdf, .md and .txt files in {}/corpus/, then run `localrag index`.", report.root.display());
    }
    if let Some(gitignore) = report.gitignore_suggestion {
        if confirm(&format!("Add .localrag/ to {}?", gitignore.display()), true)? {
            workspace::add_to_gitignore(&gitignore)?;
            println!("  Added .localrag/ to {}", gitignore.display());
        }
    }
    Ok(())
}

fn index_cmd() -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let mut conn = db::open(&ws.db_path())?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let parsers = Parsers::new(&ws);
    let report = index::sync(&ws, &mut conn, &embedder, &parsers, &mut |e| {
        let (mark, note) = match &e.change {
            Change::New => ("+", format!("{} chunks", e.chunks)),
            Change::Changed => ("~", format!("{} chunks", e.chunks)),
            Change::Touched => ("=", "unchanged content".to_string()),
            Change::Removed => ("-", "removed; kept for citations, `localrag forget` deletes it".to_string()),
            Change::Restored { from } if *from != e.path => ("↺", format!("restored from {from}")),
            Change::Restored { .. } => ("↺", "restored".to_string()),
            Change::Failed(msg) => ("!", format!("failed: {msg}")),
        };
        let ocr = if e.status == index::Status::NeedsOcr { " · needs OCR" } else { "" };
        println!("{mark} {}  ({note}{ocr})", e.path);
    })?;

    if report.reembedded > 0 {
        println!("Re-embedded {} chunks for embedding model {}", report.reembedded, ws.config.embed_model);
    }
    for (path, size) in &report.too_large {
        println!("skipped {path}: {:.0} MB exceeds max_file_mb = {}", *size as f64 / 1e6, ws.config.max_file_mb);
    }
    if !report.changed_any() {
        println!("Up to date: {} files unchanged ({} ms)", report.unchanged, report.millis);
    } else {
        println!(
            "{} new, {} changed, {} touched, {} removed, {} restored, {} failed, {} unchanged · {} chunks embedded, {} reused · {} ms",
            report.new, report.changed, report.touched, report.removed, report.restored, report.failed,
            report.unchanged, report.chunks_embedded, report.chunks_reused, report.millis
        );
    }
    if report.needs_ocr > 0 {
        println!("{} document(s) have scanned pages that need OCR (Phase 4).", report.needs_ocr);
    }
    Ok(())
}

fn status(docs: bool) -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let s = index::status(&conn)?;
    println!("Workspace   {}", ws.root.display());
    println!("Embedding   {}", s.embed_model.as_deref().unwrap_or("(not indexed yet)"));
    println!("Last sync   {}", s.last_sync.as_deref().unwrap_or("never"));
    let docs_total: i64 = s.by_status.iter().map(|(_, n)| n).sum();
    let breakdown = s.by_status.iter().map(|(k, n)| format!("{n} {k}")).collect::<Vec<_>>().join(", ");
    println!("Documents   {docs_total}{}", if breakdown.is_empty() { String::new() } else { format!(" ({breakdown})") });
    println!("Chunks      {} ({} vectors)", s.chunks, s.vectors);
    if docs {
        println!();
        for d in index::list_documents(&conn)? {
            let extra = match (d.status.as_str(), &d.error, &d.scanned_pages) {
                ("failed", Some(err), _) => format!("  {err}"),
                ("needs_ocr", _, Some(pages)) => format!("  scanned pages {pages}"),
                _ => String::new(),
            };
            println!("{:>5}  {:<10} {:>5} chunks  {}{extra}", d.id, d.status, d.chunks, d.path);
        }
    }
    Ok(())
}

fn forget(doc: Option<String>, all_removed: bool, yes: bool) -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let ids: Vec<i64> = if all_removed {
        index::list_documents(&conn)?.into_iter().filter(|d| d.status == "removed").map(|d| d.id).collect()
    } else {
        let key = doc.context("give a document id or path, or --all-removed")?;
        vec![index::find_document(&conn, &key)?.with_context(|| format!("no document matches {key:?}"))?]
    };
    if ids.is_empty() {
        println!("No removed documents.");
        return Ok(());
    }
    let what = if ids.len() == 1 { "this document".to_string() } else { format!("{} removed documents", ids.len()) };
    if !yes && !confirm(&format!("Permanently forget {what}? This cannot be undone."), false)? {
        println!("Nothing forgotten.");
        return Ok(());
    }
    for id in ids {
        let path = index::forget(&ws, &conn, id)?;
        println!("Forgot {path}");
    }
    Ok(())
}

fn run_doctor(cfg: &DoctorConfig) -> ExitCode {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let checks = runtime.block_on(doctor::run(cfg));
    for c in &checks {
        let mark = match c.status {
            doctor::Status::Pass => "ok  ",
            doctor::Status::Warn => "warn",
            doctor::Status::Fail => "FAIL",
            doctor::Status::Skip => "skip",
        };
        println!("[{mark}] {:<36} {} ({} ms)", c.name, c.detail, c.millis);
    }
    if checks.iter().any(|c| c.status == doctor::Status::Fail) { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}

/// Asks on a terminal. Without one, answers no: nothing changes without an explicit yes.
fn confirm(question: &str, default_yes: bool) -> anyhow::Result<bool> {
    if !std::io::stdin().is_terminal() {
        return Ok(false);
    }
    print!("{question} [{}] ", if default_yes { "Y/n" } else { "y/N" });
    std::io::stdout().flush()?;
    let mut answer = String::new();
    std::io::stdin().read_line(&mut answer)?;
    let a = answer.trim().to_ascii_lowercase();
    Ok(if a.is_empty() { default_yes } else { a == "y" || a == "yes" })
}
