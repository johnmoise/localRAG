//! Document embeddings via Ollama `/api/embed` (ARCHITECTURE.md §5).

use anyhow::{Context, bail};
use serde_json::{Value, json};

pub trait Embedder {
    fn model(&self) -> &str;
    /// Embeds document chunks (no instruction prefix).
    fn embed_documents(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>>;
}

pub struct OllamaEmbedder {
    host: String,
    model: String,
    client: reqwest::blocking::Client,
}

const BATCH: usize = 16;

impl OllamaEmbedder {
    pub fn new(host: &str, model: &str) -> anyhow::Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()?;
        Ok(OllamaEmbedder { host: host.trim_end_matches('/').to_string(), model: model.to_string(), client })
    }
}

impl Embedder for OllamaEmbedder {
    fn model(&self) -> &str {
        &self.model
    }

    fn embed_documents(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let resp = self
                .client
                .post(format!("{}/api/embed", self.host))
                .json(&json!({ "model": self.model, "input": batch }))
                .send()
                .with_context(|| format!("no response from Ollama at {}; is it running?", self.host))?;
            let status = resp.status();
            let v: Value = resp.json().context("Ollama /api/embed response was not JSON")?;
            if !status.is_success() {
                bail!("Ollama /api/embed ({}): {}", self.model, v["error"].as_str().unwrap_or("unknown error"));
            }
            let vecs: Vec<Vec<f32>> = serde_json::from_value(v["embeddings"].clone())?;
            if vecs.len() != batch.len() {
                bail!("Ollama returned {} embeddings for {} inputs", vecs.len(), batch.len());
            }
            out.extend(vecs);
        }
        Ok(out)
    }
}
