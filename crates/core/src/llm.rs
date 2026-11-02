//! Ollama chat and model metadata (ARCHITECTURE.md §10, §11).

use std::io::{BufRead, BufReader};

use anyhow::{Context, bail};
use serde::Serialize;
use serde_json::{Value, json};

#[derive(Debug, Clone, Default, Serialize, PartialEq)]
pub struct Message {
    pub role: String,
    pub content: String,
    /// Tool calls an assistant message made (sent back so the model sees its own calls).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    /// For a `tool` message: which tool's result this is.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_name: Option<String>,
}

/// A function call requested by the model (Ollama tool calling).
#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct ToolCall {
    pub function: ToolFunction,
}

#[derive(Debug, Clone, PartialEq, Serialize, serde::Deserialize)]
pub struct ToolFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

impl Message {
    pub fn system(c: impl Into<String>) -> Self {
        Message { role: "system".into(), content: c.into(), ..Message::default() }
    }
    pub fn user(c: impl Into<String>) -> Self {
        Message { role: "user".into(), content: c.into(), ..Message::default() }
    }
    pub fn assistant(c: impl Into<String>) -> Self {
        Message { role: "assistant".into(), content: c.into(), ..Message::default() }
    }
    /// A tool's result, answering one of the assistant's tool calls.
    pub fn tool(name: &str, c: impl Into<String>) -> Self {
        Message { role: "tool".into(), content: c.into(), tool_name: Some(name.into()), ..Message::default() }
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub model: String,
    pub messages: Vec<Message>,
    pub num_ctx: u32,
    pub temperature: f32,
    pub think: bool,
    pub num_predict: Option<i64>,
    /// Ollama structured output: a JSON schema the reply must follow.
    pub format: Option<Value>,
    /// Tool definitions (Ollama `tools`); calls come back in `ChatResponse::tool_calls`.
    pub tools: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct ChatResponse {
    pub content: String,
    pub thinking: String,
    /// Prompt tokens Ollama actually evaluated; less than sent means it truncated (§11).
    pub prompt_eval_count: Option<u64>,
    pub eval_count: Option<u64>,
    pub tool_calls: Vec<ToolCall>,
}

pub enum Delta<'a> {
    Thinking(&'a str),
    Content(&'a str),
}

#[derive(Debug, Clone, Default)]
pub struct ModelInfo {
    pub name: String,
    /// Ollama's model family, e.g. "qwen35"; also names the bundled tokenizer file.
    pub family: String,
    /// The model's advertised maximum context.
    pub context_length: Option<u32>,
    pub capabilities: Vec<String>,
}

impl ModelInfo {
    pub fn can(&self, capability: &str) -> bool {
        self.capabilities.iter().any(|c| c == capability)
    }
}

pub trait Llm {
    fn chat(&self, req: &ChatRequest, on_delta: &mut dyn FnMut(Delta)) -> anyhow::Result<ChatResponse>;
    fn model_info(&self, model: &str) -> anyhow::Result<ModelInfo>;
}

pub struct Ollama {
    host: String,
    client: reqwest::blocking::Client,
}

impl Ollama {
    pub fn new(host: &str) -> anyhow::Result<Self> {
        let client = reqwest::blocking::Client::builder().timeout(std::time::Duration::from_secs(900)).build()?;
        Ok(Ollama { host: host.trim_end_matches('/').to_string(), client })
    }

    /// Ollama's version, with a short timeout: used to show whether Ollama is reachable.
    pub fn version(&self) -> anyhow::Result<String> {
        let v: Value = self
            .client
            .get(format!("{}/api/version", self.host))
            .timeout(std::time::Duration::from_secs(2))
            .send()?
            .json()?;
        Ok(v["version"].as_str().unwrap_or("?").to_string())
    }

    /// Installed models that can chat (capability `completion`), excluding embedding-only ones.
    pub fn chat_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let tags: Value = self.client.get(format!("{}/api/tags", self.host)).send()?.json()?;
        let mut out = Vec::new();
        for m in tags["models"].as_array().into_iter().flatten() {
            let Some(name) = m["name"].as_str() else { continue };
            let info = self.model_info(name)?;
            if info.can("completion") {
                out.push(info);
            }
        }
        out.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(out)
    }
}

impl Llm for Ollama {
    fn chat(&self, req: &ChatRequest, on_delta: &mut dyn FnMut(Delta)) -> anyhow::Result<ChatResponse> {
        let mut options = json!({ "num_ctx": req.num_ctx, "temperature": req.temperature });
        if let Some(n) = req.num_predict {
            options["num_predict"] = json!(n);
        }
        let resp = self
            .client
            .post(format!("{}/api/chat", self.host))
            .json(&{
                let mut body = json!({ "model": req.model, "messages": req.messages, "stream": true, "think": req.think, "options": options });
                if let Some(f) = &req.format {
                    body["format"] = f.clone();
                }
                if let Some(t) = &req.tools {
                    body["tools"] = t.clone();
                }
                body
            })
            .send()
            .with_context(|| format!("no response from Ollama at {}; is it running?", self.host))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let v: Value = resp.json().unwrap_or_default();
            bail!("Ollama chat with {} failed ({status}): {}", req.model, v["error"].as_str().unwrap_or("unknown error"));
        }
        let mut out = ChatResponse::default();
        for line in BufReader::new(resp).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let v: Value = serde_json::from_str(&line).context("bad chat stream line")?;
            if let Some(e) = v["error"].as_str() {
                bail!("Ollama: {e}");
            }
            if let Some(t) = v["message"]["thinking"].as_str().filter(|t| !t.is_empty()) {
                out.thinking.push_str(t);
                on_delta(Delta::Thinking(t));
            }
            if let Some(t) = v["message"]["content"].as_str().filter(|t| !t.is_empty()) {
                out.content.push_str(t);
                on_delta(Delta::Content(t));
            }
            if let Some(calls) = v["message"]["tool_calls"].as_array() {
                out.tool_calls.extend(calls.iter().filter_map(|c| serde_json::from_value::<ToolCall>(c.clone()).ok()));
            }
            if v["done"].as_bool() == Some(true) {
                out.prompt_eval_count = v["prompt_eval_count"].as_u64();
                out.eval_count = v["eval_count"].as_u64();
            }
        }
        Ok(out)
    }

    fn model_info(&self, model: &str) -> anyhow::Result<ModelInfo> {
        let resp = self
            .client
            .post(format!("{}/api/show", self.host))
            .json(&json!({ "model": model }))
            .send()
            .with_context(|| format!("no response from Ollama at {}; is it running?", self.host))?;
        let status = resp.status();
        let v: Value = resp.json()?;
        if !status.is_success() {
            bail!("model {model}: {}", v["error"].as_str().unwrap_or("not found"));
        }
        let family = v["details"]["family"].as_str().unwrap_or_default().to_string();
        let context_length = v["model_info"]
            .as_object()
            .and_then(|m| m.iter().find(|(k, _)| k.ends_with(".context_length")))
            .and_then(|(_, v)| v.as_u64())
            .map(|n| n.min(u32::MAX as u64) as u32);
        let capabilities = v["capabilities"]
            .as_array()
            .map(|a| a.iter().filter_map(|c| c.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        Ok(ModelInfo { name: model.to_string(), family, context_length, capabilities })
    }
}
