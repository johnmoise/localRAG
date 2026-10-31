use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use localrag_core::doctor::{self, DoctorConfig, Status};

#[derive(Parser)]
#[command(name = "localrag", version, about = "Grounded, cited Q&A over your own literature")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
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
        /// Path to libpdfium.dylib
        #[arg(long, default_value = "third_party/pdfium/lib/libpdfium.dylib")]
        pdfium: PathBuf,
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

#[tokio::main]
async fn main() -> ExitCode {
    match Cli::parse().command {
        Command::Doctor {
            ollama_host, embed_model, chat_model, num_ctx, pdfium, tokenizer, crossref_email, offline,
        } => {
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
            let checks = doctor::run(&cfg).await;
            for c in &checks {
                let mark = match c.status {
                    Status::Pass => "ok  ",
                    Status::Warn => "warn",
                    Status::Fail => "FAIL",
                    Status::Skip => "skip",
                };
                println!("[{mark}] {:<36} {} ({} ms)", c.name, c.detail, c.millis);
            }
            if checks.iter().any(|c| c.status == Status::Fail) { ExitCode::FAILURE } else { ExitCode::SUCCESS }
        }
    }
}
