//! The reasoning agent (docs/ARCHITECTURE.md §12): a thinking, tool-calling model that reads
//! its brain, asks the corpus focused sub-questions through the grounded RAG engine, proposes
//! facts for its brain, and answers citing the original sources.

use std::collections::HashMap;

use regex::Regex;
use rusqlite::{Connection, params};
use serde::Serialize;
use serde_json::{Value, json};

use crate::answer::{CitationStatus, RagEngine};
use crate::brain::queue::{Admission, Gate, Proposal};
use crate::brain::{self, file::FactCitation, integrity::FactView};
use crate::citations;
use crate::db;
use crate::embed::Embedder;
use crate::llm::{ChatRequest, Delta, Llm, Message};
use crate::prompts;
use crate::retrieval::ChunkRef;
use crate::workspace::Workspace;

pub const MAX_STEPS: usize = 10;

pub struct Agent<'a> {
    pub ws: &'a Workspace,
    pub conn: &'a Connection,
    pub embedder: &'a dyn Embedder,
    pub llm: &'a dyn Llm,
    /// The reasoning model: owns the brain and drives the loop.
    pub model: &'a str,
    pub num_ctx: u32,
    /// Answers `ask_corpus` and judges admissions. Defaults to the reasoning model, so Ollama
    /// doesn't swap models at every step on a 16 GB machine.
    pub corpus_model: &'a str,
    pub corpus_num_ctx: u32,
}

pub enum AgentEvent<'a> {
    Step(usize),
    Thinking(&'a str),
    Token(&'a str),
    ToolCall { name: &'a str, arguments: &'a Value },
    ToolResult { name: &'a str, result: &'a Value },
    BrainUpdate(&'a Value),
    Warning(&'a str),
}

/// A citation the agent can use: from an `ask_corpus` answer.
#[derive(Debug, Clone, Serialize)]
pub struct AgentCitation {
    /// "C3"
    pub id: String,
    pub chunk: ChunkRef,
    pub quote: Option<String>,
    pub verified: bool,
    /// The sub-question whose answer it came from.
    pub question: String,
}

/// One source behind a marker in the final answer.
#[derive(Debug, Clone, Serialize)]
pub struct UsedSource {
    pub label: String,
    pub quote: Option<String>,
    pub chunk_id: Option<i64>,
    pub chunk_text_sha: String,
    pub doc_sha256: String,
    /// "C3" or "F12" (via that brain fact).
    pub via: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Step {
    pub tool: String,
    pub arguments: Value,
    pub summary: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentResult {
    pub run_id: i64,
    pub model: String,
    pub question: String,
    /// The final answer with markers replaced by source labels.
    pub text: String,
    /// As written by the model, with [C#] / [F#] markers.
    pub raw_text: String,
    pub sources: Vec<UsedSource>,
    /// Markers that name no citation or fact the agent was given.
    pub invalid_markers: Vec<String>,
    pub steps: Vec<Step>,
    pub proposals: Vec<Value>,
    pub warnings: Vec<String>,
    pub thinking: String,
}

fn marker_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\[((?:[CFS]-?\d+)(?:\s*[,;]\s*[CFS]-?\d+)*)\]").unwrap())
}

/// Tool arguments arrive as an object, or from some models as a JSON string.
fn args(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or(Value::Null),
        other => other.clone(),
    }
}

struct Run {
    citations: Vec<AgentCitation>,
    facts: HashMap<String, FactView>,
    steps: Vec<Step>,
    proposals: Vec<Value>,
    warnings: Vec<String>,
}

impl Agent<'_> {
    pub fn run(&self, question: &str, on: &mut dyn FnMut(AgentEvent)) -> anyhow::Result<AgentResult> {
        self.conn.execute("INSERT INTO agent_runs (model, question, status, created_at) VALUES (?1, ?2, 'running', ?3)", params![self.model, question, db::now_rfc3339()])?;
        let run_id = self.conn.last_insert_rowid();
        let result = self.run_inner(run_id, question, on);
        match &result {
            Ok(r) => self.conn.execute("UPDATE agent_runs SET status = 'done', result_json = ?2, finished_at = ?3 WHERE id = ?1", params![run_id, serde_json::to_string(r)?, db::now_rfc3339()])?,
            Err(e) => self.conn.execute("UPDATE agent_runs SET status = 'failed', error = ?2, finished_at = ?3 WHERE id = ?1", params![run_id, format!("{e:#}"), db::now_rfc3339()])?,
        };
        result
    }

    fn run_inner(&self, run_id: i64, question: &str, on: &mut dyn FnMut(AgentEvent)) -> anyhow::Result<AgentResult> {
        let info = self.llm.model_info(self.model)?;
        let think = info.can("thinking");
        let mut run = Run { citations: Vec::new(), facts: HashMap::new(), steps: Vec::new(), proposals: Vec::new(), warnings: Vec::new() };
        let mut messages = vec![Message::system(prompts::AGENT_SYSTEM), Message::user(question)];
        let tools = prompts::agent_tools();
        let mut thinking = String::new();
        let mut final_text: Option<String> = None;
        for step in 1..=MAX_STEPS + 1 {
            on(AgentEvent::Step(step));
            let last = step > MAX_STEPS;
            if last {
                messages.push(Message::user("You have used all your steps. Give your final answer now, citing [C#] / [F#] only."));
            }
            let req = ChatRequest {
                model: self.model.to_string(),
                messages: messages.clone(),
                num_ctx: self.num_ctx,
                temperature: 0.3,
                think,
                num_predict: None,
                format: None,
                tools: if last { None } else { Some(tools.clone()) },
            };
            let resp = self.llm.chat(&req, &mut |d| match d {
                Delta::Thinking(t) => on(AgentEvent::Thinking(t)),
                Delta::Content(t) => on(AgentEvent::Token(t)),
            })?;
            thinking.push_str(&resp.thinking);
            if resp.tool_calls.is_empty() {
                final_text = Some(resp.content);
                break;
            }
            messages.push(Message { role: "assistant".into(), content: resp.content.clone(), tool_calls: resp.tool_calls.clone(), tool_name: None });
            for call in &resp.tool_calls {
                let a = args(&call.function.arguments);
                on(AgentEvent::ToolCall { name: &call.function.name, arguments: &a });
                let (text, data) = match self.tool(run_id, &call.function.name, &a, question, &mut run, on) {
                    Ok(r) => r,
                    Err(e) => (format!("Error: {e:#}"), json!({ "error": format!("{e:#}") })),
                };
                on(AgentEvent::ToolResult { name: &call.function.name, result: &data });
                run.steps.push(Step { tool: call.function.name.clone(), arguments: a, summary: data["summary"].as_str().unwrap_or_default().to_string() });
                messages.push(Message::tool(&call.function.name, text));
            }
        }
        let raw_text = final_text.unwrap_or_default();
        let (text, sources, invalid_markers) = self.render(&raw_text, &run);
        Ok(AgentResult {
            run_id,
            model: self.model.to_string(),
            question: question.to_string(),
            text,
            raw_text,
            sources,
            invalid_markers,
            steps: run.steps,
            proposals: run.proposals,
            warnings: run.warnings,
            thinking,
        })
    }

    fn tool(&self, run_id: i64, name: &str, a: &Value, question: &str, run: &mut Run, on: &mut dyn FnMut(AgentEvent)) -> anyhow::Result<(String, Value)> {
        match name {
            "read_brain" => {
                let query = a["query"].as_str().filter(|q| !q.trim().is_empty()).unwrap_or(question);
                let (_, checked) = brain::load_checked(self.ws, self.conn, self.model)?;
                let read = brain::read(self.conn, self.embedder, &checked, query, self.num_ctx as usize / 4)?;
                for f in checked.usable() {
                    if read.included.contains(&f.id) {
                        run.facts.insert(brain::short_id(&f.id), f.clone());
                    }
                }
                for w in &read.warnings {
                    on(AgentEvent::Warning(w));
                    run.warnings.push(w.clone());
                }
                let summary = format!("{} fact(s)", read.included.len());
                Ok((read.text.clone(), json!({ "summary": summary, "facts": read.included })))
            }
            "ask_corpus" => {
                let q = a["question"].as_str().map(str::trim).filter(|q| !q.is_empty()).ok_or_else(|| anyhow::anyhow!("ask_corpus needs a question"))?;
                let engine = RagEngine { ws: self.ws, conn: self.conn, embedder: self.embedder, llm: self.llm };
                let answer = engine.ask(q, self.corpus_model, self.corpus_num_ctx, &[], &mut |_| {})?;
                // Number this answer's citations for the run; each [S#] becomes its [C#].
                let markers = citations::parse_markers(&answer.text);
                let mut text = String::new();
                let mut i = 0;
                let mut ordinal = 0;
                let mut listed = Vec::new();
                for m in &markers {
                    text.push_str(&answer.text[i..m.range.start]);
                    let mut ids = Vec::new();
                    for _ in &m.items {
                        if let Some(c) = answer.citations.iter().find(|c| c.ordinal == ordinal) {
                            let existing = run.citations.iter().find(|x| x.chunk.id == c.chunk.id && x.quote == c.quote).map(|x| x.id.clone());
                            let id = existing.unwrap_or_else(|| {
                                let id = format!("C{}", run.citations.len() + 1);
                                run.citations.push(AgentCitation {
                                    id: id.clone(),
                                    chunk: c.chunk.clone(),
                                    quote: c.quote.clone(),
                                    verified: matches!(c.status, CitationStatus::Verified | CitationStatus::VerifiedParaphrase),
                                    question: q.to_string(),
                                });
                                id
                            });
                            if !listed.contains(&id) {
                                listed.push(id.clone());
                            }
                            ids.push(id);
                        }
                        ordinal += 1;
                    }
                    text.push_str(&format!("[{}]", ids.join(", ")));
                    i = m.range.end;
                }
                text.push_str(&answer.text[i..]);
                if answer.not_found || !answer.supported {
                    let out = format!("The collection doesn't answer this well (no supported answer).\n{text}");
                    return Ok((out, json!({ "summary": "not answered by the corpus", "question": q, "supported": false })));
                }
                let mut out = format!("Answer: {text}\n\nCitations:\n");
                for id in &listed {
                    let c = run.citations.iter().find(|c| &c.id == id).expect("listed");
                    let quote = c.quote.as_deref().map(|q| format!(" \"{q}\"")).unwrap_or_default();
                    out.push_str(&format!("[{id}] {}{quote}{}\n", c.chunk.citation(), if c.verified { "" } else { " (quote not verified: can't support a fact)" }));
                }
                Ok((out, json!({ "summary": format!("{} citation(s)", listed.len()), "question": q, "supported": true, "citations": listed, "answer": answer.rendered() })))
            }
            "propose_fact" => {
                let claim = a["claim"].as_str().unwrap_or_default().trim().to_string();
                let reason = a["reason"].as_str().unwrap_or_default().trim().to_string();
                let ids: Vec<String> = a["citations"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).map(|s| s.trim().trim_matches(['[', ']']).to_uppercase()).collect();
                let mut cites: Vec<FactCitation> = Vec::new();
                let mut sub_q = None;
                let mut unknown = Vec::new();
                for id in &ids {
                    match run.citations.iter().find(|c| &c.id == id) {
                        Some(c) if c.verified && c.quote.is_some() => {
                            sub_q.get_or_insert_with(|| c.question.clone());
                            cites.push(FactCitation { label: c.chunk.citation(), quote: c.quote.clone().unwrap_or_default(), doc_sha256: c.chunk.doc_sha256.clone(), chunk_text_sha: c.chunk.text_sha.clone() });
                        }
                        Some(_) => unknown.push(format!("{id} (its quote isn't verified)")),
                        None => unknown.push(id.clone()),
                    }
                }
                let admission = if !unknown.is_empty() {
                    Admission::Dropped { rule: "cited".into(), reason: format!("unusable citation(s): {}", unknown.join(", ")) }
                } else {
                    let gate = Gate { ws: self.ws, conn: self.conn, embedder: self.embedder, llm: self.llm, model: self.model, judge_model: self.corpus_model, num_ctx: self.corpus_num_ctx };
                    gate.propose(&Proposal { claim: claim.clone(), citations: cites, reason, question: sub_q }, Some(run_id))?
                };
                let v = json!({ "claim": claim, "admission": admission });
                on(AgentEvent::BrainUpdate(&v));
                run.proposals.push(v.clone());
                let text = match &admission {
                    Admission::Queued { .. } => "Accepted into the Fact queue; the researcher will review it.".to_string(),
                    Admission::Merge { target, .. } => format!("Queued as extra support for existing fact {}.", brain::short_id(target)),
                    Admission::Dropped { reason, .. } => format!("Dropped: {reason}."),
                };
                let summary = match &admission {
                    Admission::Dropped { rule, .. } => format!("dropped ({rule})"),
                    _ => "queued".to_string(),
                };
                Ok((text, json!({ "summary": summary, "admission": admission })))
            }
            other => anyhow::bail!("there is no tool named {other}"),
        }
    }

    /// Replaces [C#] / [F#] markers with source labels. A fact is never a source itself: its
    /// original citations are shown instead.
    fn render(&self, raw: &str, run: &Run) -> (String, Vec<UsedSource>, Vec<String>) {
        let mut sources: Vec<UsedSource> = Vec::new();
        let mut invalid = Vec::new();
        let text = marker_re()
            .replace_all(raw, |caps: &regex::Captures| {
                let mut labels = Vec::new();
                for item in caps[1].split([',', ';']).map(|s| s.trim().to_uppercase()) {
                    if let Some(c) = run.citations.iter().find(|c| c.id == item) {
                        labels.push(c.chunk.citation());
                        sources.push(UsedSource { label: c.chunk.citation(), quote: c.quote.clone(), chunk_id: Some(c.chunk.id), chunk_text_sha: c.chunk.text_sha.clone(), doc_sha256: c.chunk.doc_sha256.clone(), via: c.id.clone() });
                    } else if let Some(f) = brain::long_id(&item).map(|l| brain::short_id(&l)).and_then(|s| run.facts.get(&s)) {
                        for c in &f.citations {
                            labels.push(c.label.clone());
                            sources.push(UsedSource { label: c.label.clone(), quote: Some(c.quote.clone()), chunk_id: c.chunk_id, chunk_text_sha: c.chunk_text_sha.clone(), doc_sha256: c.doc_sha256.clone(), via: brain::short_id(&f.id) });
                        }
                    } else {
                        invalid.push(item.clone());
                        labels.push(format!("{item}?"));
                    }
                }
                format!("[{}]", labels.join("; "))
            })
            .to_string();
        (text, sources, invalid)
    }
}
