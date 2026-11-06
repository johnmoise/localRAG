use std::io::{IsTerminal, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};
use localrag_core::doctor::{self, DoctorConfig};
use localrag_core::embed::OllamaEmbedder;
use localrag_core::index::{self, Change, Parsers};
use localrag_core::workspace::Workspace;
use localrag_core::answer::{Answer, Event, RagEngine, SpanSource};
use localrag_core::budget::{self, OverBudget};
use localrag_core::db;
use localrag_core::llm::Ollama;
use localrag_core::{eval, history, instance, metadata, refs, retrieval};

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
    /// Index new and changed files in corpus/ (unchanged files are skipped), then look up
    /// citation metadata on Crossref for new documents
    Index {
        /// Skip the Crossref lookups
        #[arg(long)]
        no_metadata: bool,
    },
    /// Recognize the text of scanned PDF pages with Apple Vision, then reindex those documents
    Ocr,
    /// Look up citation metadata on Crossref for documents that still need it
    Metadata {
        /// Re-query every document (bypassing the cache), except manually edited ones
        #[arg(long)]
        refresh: bool,
    },
    /// Show or change machine-wide settings (e.g. `config set crossref_email you@example.org`)
    Config {
        #[command(subcommand)]
        action: ConfigAction,
    },
    /// Show index counts, or every document with --docs
    Status {
        #[arg(long)]
        docs: bool,
    },
    /// Search the corpus (retrieval only, no model)
    Search {
        query: String,
        /// Number of results
        #[arg(short, default_value_t = 10)]
        k: usize,
    },
    /// Ask a question; the answer is grounded in and cited from the corpus
    Ask {
        question: String,
        /// Ollama chat model (default: chat_model in the workspace config)
        #[arg(long)]
        model: Option<String>,
        /// Continue the most recent chat session
        #[arg(long, short = 'c', conflicts_with = "session")]
        r#continue: bool,
        /// Continue a specific chat session
        #[arg(long)]
        session: Option<i64>,
        /// Print the full answer as JSON instead of streaming text
        #[arg(long)]
        json: bool,
    },
    /// Run the web app on 127.0.0.1 and open it (starts on the workspace picker unless -C is
    /// given). If localrag is already running, opens that instead
    Serve {
        /// Preferred port; a free one is used if it's taken
        #[arg(long, default_value_t = 8765)]
        port: u16,
        /// Don't open the browser
        #[arg(long)]
        no_open: bool,
        /// Workspace directory (default: found from the current directory)
        #[arg(long, short = 'C')]
        dir: Option<PathBuf>,
    },
    /// Check that every citation in chat history, every brain and the Fact queue still
    /// resolve (exits non-zero if anything is broken)
    Check,
    /// Research a question with a reasoning model: it reads its brain, asks the corpus
    /// sub-questions and proposes facts for its brain (approve them with `brain approve`)
    Agent {
        question: String,
        /// Reasoning model (default: reasoning_model in the workspace config)
        #[arg(long)]
        model: Option<String>,
        /// Write a 4–5 paragraph cited report instead: organizes the brain first, reads all of
        /// it (headings and thoughts too), and proposes no facts
        #[arg(long)]
        report: bool,
    },
    /// Inspect and manage a reasoning model's second brain
    Brain {
        #[command(subcommand)]
        action: BrainAction,
        /// Reasoning model (default: reasoning_model in the workspace config)
        #[arg(long, global = true)]
        model: Option<String>,
    },
    /// Measure retrieval recall and answer grounding against .localrag/eval.jsonl
    Eval {
        /// Only measure retrieval (no model calls)
        #[arg(long)]
        retrieval_only: bool,
        #[arg(long)]
        model: Option<String>,
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
        /// Default: ollama_host in the global config (http://127.0.0.1:11434)
        #[arg(long)]
        ollama_host: Option<String>,
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

#[derive(Subcommand)]
enum BrainAction {
    /// Print the brain's facts and integrity problems
    Show,
    /// List proposed facts waiting for review
    Queue,
    /// Approve a proposed fact (optionally with an edited claim)
    Approve { id: i64, #[arg(long)] claim: Option<String> },
    /// Reject a proposed fact (it won't be proposed again)
    Reject { id: i64 },
    /// Let the model reorganize its brain (fact lines are never changed)
    Organize,
    /// Undo the last reorganization
    Undo,
}

#[derive(Subcommand)]
enum ConfigAction {
    /// Print all settings
    Get,
    /// Set a setting: crossref_email, ollama_host, cache_refresh_interval_days
    Set { key: String, value: String },
    /// Clear a setting
    Unset { key: String },
}

fn main() -> ExitCode {
    let result = match Cli::parse().command {
        Command::Init { dir } => init(dir),
        Command::Index { no_metadata } => exclusive().and_then(|_g| index_cmd(!no_metadata)),
        Command::Metadata { refresh } => exclusive().and_then(|_g| metadata_cmd(refresh)),
        Command::Ocr => exclusive().and_then(|_g| ocr_cmd()),
        Command::Config { action } => config_cmd(action),
        Command::Status { docs } => status(docs),
        Command::Forget { doc, all_removed, yes } => exclusive().and_then(|_g| forget(doc, all_removed, yes)),
        Command::Search { query, k } => search(&query, k),
        Command::Ask { question, model, r#continue, session, json } => {
            exclusive().and_then(|_g| ask(&question, model, r#continue, session, json))
        }
        Command::Serve { port, no_open, dir } => serve(port, no_open, dir),
        Command::Check => check(),
        Command::Agent { question, model, report } => exclusive().and_then(|_g| agent_cmd(&question, model, report)),
        Command::Brain { action, model } => match action {
            BrainAction::Show | BrainAction::Queue => brain_cmd(action, model),
            _ => exclusive().and_then(|_g| brain_cmd(action, model)),
        },
        Command::Eval { retrieval_only, model } => exclusive().and_then(|_g| eval_cmd(retrieval_only, model)),
        Command::Doctor { ollama_host, embed_model, chat_model, num_ctx, pdfium, tokenizer, crossref_email, offline } => {
            let ollama_host = match ollama_host {
                Some(h) => h,
                None => localrag_core::config::load_global().map(|g| g.ollama_host).unwrap_or_else(|_| "http://127.0.0.1:11434".into()),
            };
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
        Err(e) if e.downcast_ref::<OverBudget>().is_some() => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
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
    println!("  .localrag/ (index, chat history, settings) is git-ignored automatically.");
    localrag_core::registry::remember(&report.root)?;
    Ok(())
}

fn index_cmd(metadata: bool) -> anyhow::Result<()> {
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
        println!("{} document(s) have scanned pages that need OCR: `localrag ocr` or Run OCR on the Documents tab.", report.needs_ocr);
    }
    if metadata {
        lookup_metadata(&ws, &conn, false)?;
    }
    Ok(())
}

fn ocr_cmd() -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let mut conn = db::open(&ws.db_path())?;
    let parsers = Parsers::new(&ws);
    let waiting = index::needing_ocr(&conn)?;
    if waiting.is_empty() {
        println!("No scanned pages waiting for OCR.");
        return Ok(());
    }
    let cancel = std::sync::atomic::AtomicBool::new(false);
    for (id, path, sha, pages) in &waiting {
        let mut last = usize::MAX;
        localrag_core::ocr::ocr_document(&ws, parsers.pdf_helper()?, &ws.corpus_dir().join(path), sha, pages, &cancel, &mut |done, total| {
            if done != last {
                print!("\rOCR {path}: page {} of {total}", (done + 1).min(total));
                let _ = std::io::stdout().flush();
                last = done;
            }
        })?;
        println!();
        index::mark_for_reprocess(&conn, *id)?;
    }
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let r = index::sync(&ws, &mut conn, &embedder, &parsers, &mut |e| println!("~ {} ({} chunks, {})", e.path, e.chunks, e.status.as_str()))?;
    println!("Reindexed {} document(s) with their OCR text.", r.changed);
    Ok(())
}

fn metadata_cmd(refresh: bool) -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    lookup_metadata(&ws, &conn, refresh)
}

fn lookup_metadata(ws: &Workspace, conn: &rusqlite::Connection, refresh: bool) -> anyhow::Result<()> {
    let client = metadata::crossref::configured(&ws.global);
    if refresh {
        conn.execute("UPDATE documents SET metadata_status = 'pending' WHERE metadata_source IS NOT 'manual'", [])?;
    }
    let pending: i64 = conn.query_row(
        "SELECT count(*) FROM documents WHERE metadata_status = 'pending' AND status IN ('indexed', 'needs_ocr')", [], |r| r.get(0))?;
    if pending == 0 {
        return lookup_references(ws, conn, client);
    }
    let pool = match client.pool() {
        metadata::crossref::Pool::Polite => "polite pool",
        metadata::crossref::Pool::Public => "public pool; set an email with `localrag config set crossref_email …` for faster lookups",
    };
    println!("Looking up metadata for {pending} document(s) on Crossref ({pool})…");
    let r = metadata::process_pending(conn, client, &mut |e| {
        let mark = match e.status.as_str() { "matched" => "✓", "proposed" => "?", _ => "·" };
        println!("{mark} {}  ({})", e.path, e.detail);
    })?;
    println!("{} matched, {} need confirming (Documents tab), {} without a match", r.matched, r.proposed, r.unmatched);
    for e in &r.errors {
        eprintln!("warning: {e} (will retry next time)");
    }
    lookup_references(ws, conn, client)
}

fn lookup_references(ws: &Workspace, conn: &rusqlite::Connection, client: &metadata::crossref::Crossref) -> anyhow::Result<()> {
    let _ = localrag_core::references::backfill_cited_by(conn, client);
    let pending: i64 = conn.query_row("SELECT count(*) FROM doc_references WHERE status = 'pending'", [], |r| r.get(0))?;
    if pending > 0 {
        enrich_references(conn, client, pending)?;
    }
    if let Err(e) = localrag_core::references::backfill_details(conn, client, &mut |_, _| {}) {
        eprintln!("warning: {e:#} (abstracts will be fetched next time)");
    }
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let r = localrag_core::references::embed_pending(conn, &embedder)?;
    if r.embedded + r.reused > 0 {
        println!("  {} reference chunk(s) embedded, {} reused", r.embedded, r.reused);
    }
    Ok(())
}

fn enrich_references(conn: &rusqlite::Connection, client: &metadata::crossref::Crossref, pending: i64) -> anyhow::Result<()> {
    println!("Looking up {pending} bibliography reference(s) on Crossref…");
    let mut last = usize::MAX;
    let r = localrag_core::references::enrich_pending(conn, client, &mut |done, total| {
        if done / 10 != last {
            last = done / 10;
            print!("\r  {done} of {total}");
            let _ = std::io::stdout().flush();
        }
    })?;
    println!("\r  {} matched on Crossref, {} kept as stubs from the citation text", r.matched, r.stubs);
    for e in &r.errors {
        eprintln!("warning: {e} (will retry next time)");
    }
    Ok(())
}

fn config_cmd(action: ConfigAction) -> anyhow::Result<()> {
    let mut g = localrag_core::config::load_global()?;
    match action {
        ConfigAction::Get => {
            println!("crossref_email = {}", g.crossref_email.as_deref().unwrap_or("(not set: Crossref public pool)"));
            println!("ollama_host    = {}", g.ollama_host);
            println!("cache_refresh_interval_days = {} (Crossref responses are reused for this long)", g.cache_refresh_interval_days);
            println!("(stored in {}, private to your user)", localrag_core::config::global_config_dir().join("config.toml").display());
            return Ok(());
        }
        ConfigAction::Set { key, value } => match key.as_str() {
            "crossref_email" => g.crossref_email = Some(localrag_core::config::validate_email(&value)?),
            "ollama_host" => g.ollama_host = value,
            "cache_refresh_interval_days" => g.cache_refresh_interval_days = localrag_core::config::validate_cache_days(&value)?,
            _ => anyhow::bail!("unknown setting {key:?}; known: crossref_email, ollama_host, cache_refresh_interval_days"),
        },
        ConfigAction::Unset { key } => match key.as_str() {
            "crossref_email" => g.crossref_email = None,
            "cache_refresh_interval_days" => g.cache_refresh_interval_days = localrag_core::metadata::crossref::DEFAULT_REFRESH_DAYS,
            _ => anyhow::bail!("{key:?} can't be unset"),
        },
    }
    localrag_core::config::save_global(&g)?;
    println!("Saved.");
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

fn search(query: &str, k: usize) -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let opts = retrieval::SearchOptions {
        candidates: ws.config.retrieval.candidates,
        limit: k,
        per_doc_cap: ws.config.retrieval.per_doc_cap,
        ..retrieval::SearchOptions::default()
    };
    let hits = retrieval::search(&conn, &embedder, query, &opts)?;
    if hits.is_empty() {
        println!("No matches.");
    }
    for (i, h) in hits.iter().enumerate() {
        let sim = h.similarity.map_or("   –".to_string(), |s| format!("{s:.2}"));
        let ranks = format!(
            "bm25 {} · vec {}",
            h.bm25_rank.map_or("–".into(), |r| r.to_string()),
            h.vec_rank.map_or("–".into(), |r| r.to_string())
        );
        println!("{:>2}. [{}]  sim {sim} · {ranks}", i + 1, h.chunk.citation());
        println!("    {}", snippet(&h.chunk.text, 240));
    }
    Ok(())
}

fn ask(question: &str, model: Option<String>, cont: bool, session: Option<i64>, json: bool) -> anyhow::Result<()> {
    let mut ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let llm = Ollama::new(&ws.global.ollama_host)?;
    let model = model.unwrap_or_else(|| ws.config.chat_model.clone());
    let num_ctx = resolve_num_ctx(&mut ws, &llm, &model)?;

    let session = match (session, cont) {
        (Some(id), _) => {
            anyhow::ensure!(history::session_exists(&conn, id)?, "no chat session {id}");
            Some(id)
        }
        (None, true) => history::latest_session(&conn)?,
        _ => None,
    };
    let turns = match session {
        Some(id) => history::turns(&conn, id)?,
        None => Vec::new(),
    };

    let engine = RagEngine { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm };
    let mut streamed = false;
    let answer = engine.ask(question, &model, num_ctx, &turns, &mut |e| match e {
        Event::Query(q) if !json => eprintln!("(searching for: {q})"),
        Event::Warning(w) => eprintln!("warning: {w}"),
        Event::Token(t) if !json => {
            streamed = true;
            print!("{t}");
            let _ = std::io::stdout().flush();
        }
        _ => {}
    })?;

    let session = match session {
        Some(id) => id,
        None => history::create_session(&conn, question, &model)?,
    };
    history::record(&conn, session, question, &answer)?;

    if json {
        println!("{}", serde_json::to_string_pretty(&answer)?);
        return Ok(());
    }
    if streamed {
        println!("\n");
    }
    print_answer(&answer, streamed);
    eprintln!("(session {session}; continue with `localrag ask -c \"…\"`)");
    Ok(())
}

fn print_answer(a: &Answer, streamed: bool) {
    if a.not_found || a.text.starts_with("INSUFFICIENT") {
        if !streamed {
            println!("{}\n", a.text);
        }
        if !a.sources.is_empty() {
            println!("Closest material found (not sufficient to answer):");
            for c in &a.sources {
                println!("  - [{}] {}", c.citation(), snippet(&c.text, 160));
            }
        }
        if !a.suggestions.is_empty() {
            println!("\nMaterial that could help (suggestions from the model's general knowledge, unverified):");
            for s in &a.suggestions {
                println!("  - {s}");
            }
        }
        return;
    }
    println!("Answer with citations:\n{}\n", a.rendered());
    if !a.citations.is_empty() {
        println!("Citations:");
        for c in &a.citations {
            let mark = match c.span_source {
                SpanSource::Quote if c.quote_repaired => "✓ quote re-selected from the source",
                SpanSource::Quote => "✓ quote found",
                SpanSource::Similar if c.quote.is_some() => "✗ quote NOT in source; closest sentence shown",
                SpanSource::Similar => "– no quote; closest sentence shown",
                SpanSource::None => "✗ no supporting text found",
            };
            let span = c.span.map(|(a, b)| c.chunk.text.chars().skip(a).take(b - a).collect::<String>()).unwrap_or_default();
            let support = match c.support {
                Some(localrag_core::answer::Support::No) => "  ✗ strict check: passage does NOT support the sentence",
                Some(localrag_core::answer::Support::Partial) => "  ~ strict check: partial support",
                Some(localrag_core::answer::Support::Yes) => "  ✓ strict check: supported",
                None => "",
            };
            println!("  [S{}] {}  {mark}{support}", c.marker, c.chunk.citation());
            if !span.is_empty() {
                println!("        \"{}\"", snippet(&span, 200));
            }
        }
    }
    if !a.invalid_markers.is_empty() {
        println!("Invalid citations (no such source): {}", a.invalid_markers.join(", "));
    }
    if !a.unsupported_sentences.is_empty() {
        println!("\nSentences without a citation:");
        for s in &a.unsupported_sentences {
            println!("  ! {s}");
        }
    }
    if !a.supported {
        println!("\n⚠ This answer is NOT supported: no citation could be verified against the sources.");
    }
}

fn serve(port: u16, no_open: bool, dir: Option<PathBuf>) -> anyhow::Result<()> {
    let open_browser = |url: &str| {
        if !no_open {
            let _ = std::process::Command::new("open").arg(url).status();
        }
    };
    let guard = match instance::acquire()? {
        Ok(g) => g,
        Err(running) => {
            // Only one localrag at a time: hand over to the one that's running.
            match &running.url {
                Some(url) => {
                    println!("localrag is already running (pid {}). Opening it: {url}", running.pid);
                    open_browser(url);
                }
                None => println!("localrag is already starting (pid {}).", running.pid),
            }
            println!("Switch workspaces from its File menu.");
            return Ok(());
        }
    };
    let root = match dir {
        Some(d) => Some(Workspace::discover(&d)?.root),
        None => None,
    };
    let runtime = tokio::runtime::Runtime::new()?;
    let result = runtime.block_on(localrag_server::serve(root.as_deref(), port, |url| {
        match &root {
            Some(r) => println!("localrag is serving {}", r.display()),
            None => println!("localrag is running; choose a workspace in the browser."),
        }
        println!("  {url}");
        println!("  (press Ctrl-C to stop)");
        let _ = guard.publish_url(url);
        open_browser(url);
    }));
    // Ctrl-C: don't wait for in-flight work (an answer, a sync) before exiting.
    runtime.shutdown_background();
    result
}

/// Commands that write to a workspace refuse while the app is running, so there's only ever
/// one copy of localrag changing things.
fn exclusive() -> anyhow::Result<instance::InstanceGuard> {
    match instance::acquire()? {
        Ok(g) => Ok(g),
        Err(running) => anyhow::bail!(
            "localrag is already running (pid {}){}. Use it there, or stop it (Ctrl-C) before running this command",
            running.pid,
            running.url.map(|u| format!(" at {u}")).unwrap_or_default()
        ),
    }
}

fn check() -> anyhow::Result<()> {
    let ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let r = refs::check_history(&conn)?;
    println!(
        "Chat history: {} citations checked · {} resolved · {} relocated · {} source removed · {} broken",
        r.checked, r.resolved, r.relocated, r.source_removed, r.broken.len()
    );
    for (s, m, label, path) in &r.removed {
        println!("  removed  session {s}, message {m}: [{label}] ({path})");
    }
    for (s, m, label, path) in &r.broken {
        println!("  BROKEN   session {s}, message {m}: [{label}] ({path})");
    }
    let (brains, queue) = localrag_core::brain::check_all(&ws, &conn)?;
    let mut brain_problems = 0;
    for b in &brains {
        println!("Brain {}: {} usable fact(s), {} problem(s)", b.model, b.usable().len(), b.problems.len());
        for p in &b.problems {
            println!("  {:<16} {}", p.kind, p.message);
        }
        brain_problems += b.problems.len();
    }
    for q in &queue {
        println!("  queue  {}: {}", q.model, q.message);
    }
    anyhow::ensure!(r.ok() && brain_problems == 0 && queue.is_empty(), "{} broken citation(s), {brain_problems} brain problem(s), {} queue problem(s)", r.broken.len(), queue.len());
    Ok(())
}

fn reasoning_model(ws: &Workspace, model: Option<String>) -> anyhow::Result<String> {
    model.or_else(|| ws.config.reasoning_model.clone()).context("name a reasoning model with --model, or set reasoning_model in .localrag/config.toml")
}

fn agent_cmd(question: &str, model: Option<String>, report: bool) -> anyhow::Result<()> {
    let mut ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let llm = Ollama::new(&ws.global.ollama_host)?;
    let model = reasoning_model(&ws, model)?;
    let info = localrag_core::llm::Llm::model_info(&llm, &model)?;
    anyhow::ensure!(info.can("tools") && info.can("thinking"), "{model} isn't a reasoning model (it needs Ollama's tools and thinking capabilities)");
    let corpus_model = ws.config.agent_corpus_model.clone().unwrap_or_else(|| model.clone());
    let num_ctx = resolve_num_ctx(&mut ws, &llm, &model)?;
    let corpus_num_ctx = resolve_num_ctx(&mut ws, &llm, &corpus_model)?;
    let agent = localrag_core::agent::Agent { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm, model: &model, num_ctx, corpus_model: &corpus_model, corpus_num_ctx };
    use localrag_core::agent::AgentEvent;
    let mode = if report { localrag_core::agent::Mode::Report } else { localrag_core::agent::Mode::Research };
    let r = agent.run_mode(mode, question, &mut |e| match e {
        AgentEvent::ToolCall { name, arguments } => eprintln!("→ {name} {}", arguments),
        AgentEvent::ToolResult { name, result } => eprintln!("  {name}: {}", result["summary"].as_str().unwrap_or("")),
        AgentEvent::BrainUpdate(v) => eprintln!("  proposal: {}", v),
        AgentEvent::Warning(w) => eprintln!("warning: {w}"),
        _ => {}
    })?;
    println!("\n{}\n", r.text);
    if !r.invalid_markers.is_empty() {
        println!("Citations that don't resolve: {}", r.invalid_markers.join(", "));
    }
    let queued = r.proposals.iter().filter(|p| p["admission"]["outcome"] != "dropped").count();
    if queued > 0 {
        println!("{queued} fact(s) proposed: review them with `localrag brain queue --model {model}`.");
    }
    Ok(())
}

fn brain_cmd(action: BrainAction, model: Option<String>) -> anyhow::Result<()> {
    use localrag_core::brain::{self, integrity::LineView, organize, queue};
    let mut ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let model = reasoning_model(&ws, model)?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let llm = Ollama::new(&ws.global.ollama_host)?;
    let gate_ctx = |ws: &mut Workspace| -> anyhow::Result<(String, u32)> {
        let judge = ws.config.agent_corpus_model.clone().unwrap_or_else(|| model.clone());
        let n = resolve_num_ctx(ws, &llm, &judge)?;
        Ok((judge, n))
    };
    match action {
        BrainAction::Show => {
            let (_, c) = brain::load_checked(&ws, &conn, &model)?;
            for l in &c.lines {
                match l {
                    LineView::Heading { level, text } => println!("{} {text}", "#".repeat(*level)),
                    LineView::Thought { text, .. } => println!("  (thought) {text}"),
                    LineView::Fact(f) => println!("  {} [{}] {} ({})", f.id, f.status, f.claim, f.citations.iter().map(|c| c.label.as_str()).collect::<Vec<_>>().join("; ")),
                    LineView::Invalid { raw, .. } => println!("  (invalid) {raw}"),
                    LineView::Blank => {}
                }
            }
            for p in &c.problems {
                println!("problem: {}", p.message);
            }
        }
        BrainAction::Queue => {
            let items = queue::pending(&conn, &model)?;
            if items.is_empty() {
                println!("No proposed facts for {model}.");
            }
            for i in items {
                let merge = i.target_fact_id.as_deref().map(|t| format!(" (adds support to {t})")).unwrap_or_default();
                println!("{:>4}. {}{merge}\n      {}\n      why: {}", i.id, i.claim, i.citations.iter().map(|c| format!("{} \"{}\"", c.label, c.quote)).collect::<Vec<_>>().join("; "), i.reason.unwrap_or_default());
            }
        }
        BrainAction::Approve { id, claim } => {
            let (judge, n) = gate_ctx(&mut ws)?;
            let gate = queue::Gate { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm, model: &model, judge_model: &judge, num_ctx: n };
            println!("Approved as {}.", gate.approve(id, claim.as_deref())?);
        }
        BrainAction::Reject { id } => {
            let (judge, n) = gate_ctx(&mut ws)?;
            queue::Gate { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm, model: &model, judge_model: &judge, num_ctx: n }.reject(id)?;
            println!("Rejected.");
        }
        BrainAction::Organize => {
            let n = resolve_num_ctx(&mut ws, &llm, &model)?;
            let (_, c) = brain::load_checked(&ws, &conn, &model)?;
            let r = organize::organize(&ws, &llm, &model, n, &c)?;
            match (r.changed, r.errors.is_empty()) {
                (true, _) => println!("Reorganized (undo with `localrag brain undo`)."),
                (false, true) => println!("No changes."),
                (false, false) => println!("Not saved; the layout failed validation twice:\n- {}", r.errors.join("\n- ")),
            }
        }
        BrainAction::Undo => {
            localrag_core::brain::file::undo(&ws, &model)?;
            println!("Restored the previous version.");
        }
    }
    Ok(())
}

fn eval_cmd(retrieval_only: bool, model: Option<String>) -> anyhow::Result<()> {
    let mut ws = Workspace::discover(&std::env::current_dir()?)?;
    let conn = db::open(&ws.db_path())?;
    let path = ws.state_dir().join("eval.jsonl");
    let items = eval::load(&path)?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;

    let retrieval = eval::retrieval(&conn, &embedder, &items)?;
    println!("Retrieval ({} questions):", retrieval.len());
    for r in &retrieval {
        let rank = |r: Option<usize>| r.map_or("miss".to_string(), |n| format!("#{n}"));
        let (doc, page) = if r.answerable { (rank(r.doc_rank), rank(r.page_rank)) } else { ("–".into(), "–".into()) };
        println!("  doc {doc:>4}  page {page:>4}  sim {:.2}  {}{}", r.best_similarity.unwrap_or(0.0), if r.answerable { "" } else { "(unanswerable) " }, r.question);
    }

    let mut answers = Vec::new();
    let mut model_name = None;
    if !retrieval_only {
        let llm = Ollama::new(&ws.global.ollama_host)?;
        let model = model.unwrap_or_else(|| ws.config.chat_model.clone());
        let num_ctx = resolve_num_ctx(&mut ws, &llm, &model)?;
        let engine = RagEngine { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm };
        println!("\nAnswers ({model}, num_ctx {num_ctx}):");
        answers = eval::answers(&engine, &items, &model, num_ctx, &mut |i, r| {
            let verdict = match (r.answerable, r.supported) {
                (true, true) => "supported",
                (true, false) => "UNSUPPORTED",
                (false, false) => "refused ✓",
                (false, true) => "ANSWERED (should refuse)",
            };
            println!(
                "  {:>2}. {verdict:<24} cites {} (verified {}, re-selected {}, on doc {}) uncited {}  {}",
                i + 1, r.citations, r.verified, r.repaired, r.on_expected_doc, r.uncited_sentences, r.question
            );
        })?;
        model_name = Some(model);
    }

    let summary = eval::summarize(&items, &retrieval, &answers);
    let pct = |v: Option<f32>| v.map_or("n/a".to_string(), |v| format!("{:.0}%", v * 100.0));
    println!("\nSummary");
    println!("  doc recall@5 {}  @10 {}", pct(Some(summary.doc_recall_at_5)), pct(Some(summary.doc_recall_at_10)));
    println!("  page recall@5 {}  @10 {}", pct(summary.page_recall_at_5), pct(summary.page_recall_at_10));
    let sim = |v: Option<f32>| v.map_or("n/a".to_string(), |v| format!("{v:.2}"));
    println!(
        "  best similarity: lowest answerable {} · highest unanswerable {} · gate min_similarity {:.2}",
        sim(summary.lowest_answerable_similarity), sim(summary.highest_unanswerable_similarity), ws.config.retrieval.min_similarity
    );
    if !retrieval_only {
        println!("  answerable supported {}  ·  unanswerable refused {}", pct(summary.supported_rate), pct(summary.refusal_rate));
        println!(
            "  quote verification {} (re-selected {})  ·  citation precision {}  ·  rejected by strict check {}",
            pct(summary.quote_verification_rate), pct(summary.repaired_rate), pct(summary.citation_precision), pct(summary.rejected_by_check_rate)
        );
    }
    let out = ws.state_dir().join("logs").join(format!("eval-{}.json", db::now_rfc3339().replace(':', "")));
    std::fs::write(&out, serde_json::to_string_pretty(&serde_json::json!({
        "model": model_name, "summary": summary, "retrieval": retrieval, "answers": answers,
    }))?)?;
    println!("  details: {}", out.display());
    Ok(())
}

fn resolve_num_ctx(ws: &mut Workspace, llm: &Ollama, model: &str) -> anyhow::Result<u32> {
    let (n, new) = budget::resolve_num_ctx(ws, llm, model)?;
    if new {
        eprintln!("(using num_ctx {n} for {model}, based on this Mac's memory; change it in .localrag/config.toml)");
    }
    Ok(n)
}

fn snippet(text: &str, max: usize) -> String {
    let flat = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if flat.chars().count() <= max { flat } else { format!("{}…", flat.chars().take(max).collect::<String>()) }
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
