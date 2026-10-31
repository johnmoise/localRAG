use anyhow::{Context, bail, ensure};
use serde_json::{Value, json};

use super::{DoctorConfig, Outcome};

/// Query instruction for Qwen3-Embedding (§5). Documents are embedded without one.
pub(crate) fn embed_query(question: &str) -> String {
    format!(
        "Instruct: Given a research question, retrieve passages from the literature that answer it\nQuery: {question}"
    )
}

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
        .send()
        .await
        .with_context(|| format!("no response from {host}; is Ollama running?"))?
        .json()
        .await?;
    Ok(Outcome::Pass(format!("Ollama {} at {host}", v["version"].as_str().unwrap_or("?"))))
}

/// 1024 dimensions, and the query template ranks the relevant passage first.
pub(super) async fn check_embed(client: &reqwest::Client, cfg: &DoctorConfig) -> anyhow::Result<Outcome> {
    let docs = [
        "Self-attention takes time quadratic in the sequence length.",
        "The migratory patterns of Arctic terns span both poles.",
    ];
    let query = embed_query("How does the cost of attention scale with sequence length?");
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
