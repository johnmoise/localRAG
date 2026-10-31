use anyhow::{Context, ensure};
use serde_json::json;
use tokenizers::Tokenizer;

use super::ollama::post;
use super::{DoctorConfig, Outcome};

const SAMPLE: &str = "Self-attention takes O(n²) time and memory in sequence length n, \
    which limits context size. Linear attention (Katharopoulos et al., 2020) reduces this to O(n).";

/// Local tokenizer count vs. Ollama's `prompt_eval_count` for the same raw text (§11).
/// `raw: true` skips the chat template, so both sides count exactly the same characters.
pub(super) async fn check(client: &reqwest::Client, cfg: &DoctorConfig) -> anyhow::Result<Outcome> {
    let file = &cfg.tokenizer_file;
    if !file.exists() {
        return Ok(Outcome::Skip(format!("{} not found; run scripts/fetch-tokenizers.sh", file.display())));
    }
    let tokenizer = Tokenizer::from_file(file).map_err(|e| anyhow::anyhow!("{e}"))?;
    let local = tokenizer.encode(SAMPLE, false).map_err(|e| anyhow::anyhow!("{e}"))?.len() as i64;

    let v = post(client, &cfg.ollama_host, "/api/generate", json!({
        "model": cfg.chat_model,
        "prompt": SAMPLE,
        "raw": true,
        "stream": false,
        "think": false,
        "options": {"num_ctx": cfg.num_ctx, "num_predict": 1},
    }))
    .await?;
    let ollama = v["prompt_eval_count"].as_i64().context("no prompt_eval_count in response")?;

    let diff = ollama - local;
    ensure!(diff.abs() <= 2, "local count {local} vs Ollama {ollama} (off by {diff})");
    let note = if diff == 0 { "exact match".to_string() } else { format!("Ollama adds {diff} (BOS/special tokens)") };
    Ok(Outcome::Pass(format!("{}: local {local}, Ollama {ollama}: {note}", cfg.chat_model)))
}
