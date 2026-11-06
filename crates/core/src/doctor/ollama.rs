use anyhow::{Context, bail, ensure};
use serde_json::{Value, json};

use super::{DoctorConfig, Outcome};

pub(super) async fn post(client: &reqwest::Client, host: &str, path: &str, body: Value) -> anyhow::Result<Value> {
    let resp = client.post(format!("{host}{path}")).json(&body).send().await?;
    let status = resp.status();
    let value: Value = resp.json().await.context("Ollama response was not JSON")?;
    if !status.is_success() {
        bail!("{path} returned {status}: {}", value["error"].as_str().unwrap_or("unknown error"));
    }
    Ok(value)
}

pub(super) async fn check_reachable(client: &reqwest::Client, host: &str) -> anyhow::Result<Outcome> {
    let v: Value = client
        .get(format!("{host}/api/version"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
        .with_context(|| format!("no response from {host}; is Ollama running? (`localrag serve` starts an installed Ollama)"))?
        .json()
        .await?;
    Ok(Outcome::Pass(format!("Ollama {} at {host}", v["version"].as_str().unwrap_or("?"))))
}

/// Where Ollama is installed (what `localrag serve` would start when it's stopped, §19.1).
/// Never starts anything itself.
pub(super) fn check_installed(host: &str, reachable: bool) -> anyhow::Result<Outcome> {
    use crate::ollama::{self, SearchEnv};
    if !ollama::is_local_host(host) {
        return Ok(Outcome::Skip(format!("{host} is another machine; localrag never starts Ollama there")));
    }
    let found = SearchEnv::current().discover();
    let mut parts = Vec::new();
    if let Some(a) = &found.app {
        parts.push(format!("app {}", a.display()));
    }
    if let Some(c) = &found.cli {
        parts.push(format!("cli {}", c.display()));
    }
    match (found.found(), reachable) {
        (true, true) => Ok(Outcome::Pass(parts.join("; "))),
        (true, false) => Ok(Outcome::Warn(format!("{}; stopped: `localrag serve` starts it", parts.join("; ")))),
        (false, true) => Ok(Outcome::Pass("running, but not at a known install location".into())),
        (false, false) => bail!("Ollama isn't installed: get it from {} (macOS: `brew install ollama`; Windows: `winget install Ollama.Ollama`)", ollama::DOWNLOAD_URL),
    }
}

/// 1024 dimensions, and the query template ranks the relevant passage first.
pub(super) async fn check_embed(client: &reqwest::Client, cfg: &DoctorConfig) -> anyhow::Result<Outcome> {
    let docs = [
        "Self-attention takes time quadratic in the sequence length.",
        "The migratory patterns of Arctic terns span both poles.",
    ];
    let query = crate::embed::query_text("How does the cost of attention scale with sequence length?");
    let v = post(client, &cfg.ollama_host, "/api/embed", json!({
        "model": cfg.embed_model,
        "input": [docs[0], docs[1], query],
    }))
    .await
    .with_context(|| format!("try `ollama pull {}`", cfg.embed_model))?;

    let vecs: Vec<Vec<f32>> = serde_json::from_value(v["embeddings"].clone())?;
    let dim = vecs[0].len();
    ensure!(dim == 1024, "{} returned {dim} dimensions, expected 1024", cfg.embed_model);
    let cos = |a: &[f32], b: &[f32]| {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let norm = |x: &[f32]| x.iter().map(|v| v * v).sum::<f32>().sqrt();
        dot / (norm(a) * norm(b))
    };
    let (relevant, unrelated) = (cos(&vecs[2], &vecs[0]), cos(&vecs[2], &vecs[1]));
    ensure!(relevant > unrelated, "query matched the unrelated passage ({relevant:.3} vs {unrelated:.3})");
    Ok(Outcome::Pass(format!(
        "{}: {dim} dims; query→relevant {relevant:.3} vs unrelated {unrelated:.3}",
        cfg.embed_model
    )))
}

/// A short chat with `num_ctx` set; reports the prompt token count Ollama measured (§11).
pub(super) async fn check_chat(client: &reqwest::Client, cfg: &DoctorConfig) -> anyhow::Result<Outcome> {
    let v = post(client, &cfg.ollama_host, "/api/chat", json!({
        "model": cfg.chat_model,
        "messages": [{"role": "user", "content": "Reply with the single word: ready"}],
        "stream": false,
        "think": false,
        "options": {"num_ctx": cfg.num_ctx, "temperature": 0, "num_predict": 16},
    }))
    .await
    .with_context(|| format!("chat with {}", cfg.chat_model))?;

    let reply = v["message"]["content"].as_str().unwrap_or("").trim().to_string();
    let prompt_tokens = v["prompt_eval_count"].as_u64().unwrap_or(0);
    let loaded = loaded_context(client, &cfg.ollama_host, &cfg.chat_model).await;
    let ctx_note = match loaded {
        Some(n) if n == cfg.num_ctx as u64 => format!("loaded with context {n}"),
        Some(n) => return Ok(Outcome::Warn(format!(
            "asked for num_ctx {} but Ollama loaded {} with context {n}", cfg.num_ctx, cfg.chat_model
        ))),
        None => "loaded context not reported".into(),
    };
    Ok(Outcome::Pass(format!(
        "{}: replied {reply:?}; prompt_eval_count {prompt_tokens}; {ctx_note}",
        cfg.chat_model
    )))
}

/// Context length of a loaded model, from `/api/ps`.
async fn loaded_context(client: &reqwest::Client, host: &str, model: &str) -> Option<u64> {
    let v: Value = client.get(format!("{host}/api/ps")).send().await.ok()?.json().await.ok()?;
    v["models"].as_array()?.iter()
        .find(|m| m["name"].as_str() == Some(model) || m["model"].as_str() == Some(model))?
        ["context_length"].as_u64()
}
