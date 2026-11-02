//! Second Brain and agent routes (docs/ARCHITECTURE.md §12, §14).

use std::convert::Infallible;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

use localrag_core::agent::{Agent, AgentEvent, Mode};
use localrag_core::brain::{self, file, integrity::LineView, organize, queue};
use localrag_core::budget;
use localrag_core::db;
use localrag_core::embed::OllamaEmbedder;
use localrag_core::llm::{Llm, Ollama};
use localrag_core::workspace::Workspace;

use crate::AppState;
use crate::routes::{ApiError, ApiResult, blocking, not_found};

pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/api/brain/models", get(models))
        .route("/api/brain/{model}", get(show))
        .route("/api/brain/{model}/lines/{n}", patch(edit_line).delete(delete_line))
        .route("/api/brain/{model}/facts/{id}", patch(edit_fact).delete(delete_fact))
        .route("/api/brain/{model}/facts/{id}/restore", post(restore_fact))
        .route("/api/brain/{model}/organize", post(organize_brain))
        .route("/api/brain/{model}/undo", post(undo))
        .route("/api/brain/{model}/last-change", get(last_change))
        .route("/api/brain/{model}/queue", get(queue_list))
        .route("/api/brain/{model}/queue/{id}/{action}", post(queue_action))
        .route("/api/agent/run", post(run))
        .route("/api/agent/runs", get(runs))
        .route("/api/agent/runs/{id}", get(run_detail).delete(delete_run))
}

fn unprocessable(e: anyhow::Error) -> ApiError {
    if e.downcast_ref::<file::Conflict>().is_some() {
        return e.into();
    }
    ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "rejected", format!("{e:#}"))
}

fn if_match(h: &HeaderMap) -> Option<String> {
    h.get("if-match").and_then(|v| v.to_str().ok()).map(|s| s.trim_matches('"').to_string())
}

/// Installed models that can reason with tools (completion + thinking + tools).
async fn models(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let installed = Ollama::new(&ws.global.ollama_host)?.chat_models().unwrap_or_default();
        let reasoning: Vec<String> = installed.iter().filter(|m| m.can("thinking") && m.can("tools")).map(|m| m.name.clone()).collect();
        let with_brains = file::models_with_brains(&ws);
        let mut all: Vec<Value> = reasoning.iter().map(|m| json!({ "name": m, "installed": true, "has_brain": with_brains.contains(m) })).collect();
        for m in with_brains.iter().filter(|m| !reasoning.contains(m)) {
            all.push(json!({ "name": m, "installed": false, "has_brain": true }));
        }
        let default = ws.config.reasoning_model.clone().filter(|m| reasoning.contains(m)).or_else(|| with_brains.iter().find(|m| reasoning.contains(m)).cloned()).or_else(|| reasoning.first().cloned());
        Ok(Json(json!({ "models": all, "default": default })))
    })
    .await
}

fn brain_json(ws: &Workspace, conn: &rusqlite::Connection, model: &str) -> Result<Value, ApiError> {
    let (b, checked) = brain::load_checked(ws, conn, model)?;
    let pending = queue::pending(conn, model)?.len();
    Ok(json!({
        "model": model,
        "exists": b.exists,
        "hash": checked.hash,
        "lines": checked.lines,
        "problems": checked.problems,
        "usable": checked.usable().len(),
        "pending": pending,
        "can_undo": !file::history(ws, model).is_empty(),
        "path": b.path.strip_prefix(&ws.root).unwrap_or(&b.path).display().to_string(),
    }))
}

async fn show(State(state): State<AppState>, Path(model): Path<String>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        Ok(Json(brain_json(&ws, &conn, &model)?))
    })
    .await
}

#[derive(Deserialize)]
struct LineEdit {
    text: String,
}

/// Edits a heading or thought (facts have their own route). `If-Match`: the file hash.
async fn edit_line(State(state): State<AppState>, Path((model, n)): Path<(String, usize)>, headers: HeaderMap, Json(body): Json<LineEdit>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let text = body.text.split_whitespace().collect::<Vec<_>>().join(" ");
        file::update(&ws, &model, if_match(&headers).as_deref(), false, |lines| {
            let line = lines.get_mut(n).ok_or_else(|| anyhow::anyhow!("no line {n}"))?;
            *line = match line {
                file::Line::Thought { .. } if text.contains("[FACT") => anyhow::bail!("a thought can't contain a [FACT] marker"),
                file::Line::Thought { .. } => file::Line::Thought { text: text.clone() },
                file::Line::Heading { level, .. } => file::Line::Heading { level: *level, text: text.clone() },
                _ => anyhow::bail!("only headings and thoughts are edited here"),
            };
            if text.is_empty() {
                anyhow::bail!("empty text: delete the line instead");
            }
            Ok(())
        })
        .map_err(unprocessable)?;
        Ok(Json(brain_json(&ws, &conn, &model)?))
    })
    .await
}

async fn delete_line(State(state): State<AppState>, Path((model, n)): Path<(String, usize)>, headers: HeaderMap) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        file::update(&ws, &model, if_match(&headers).as_deref(), false, |lines| {
            match lines.get(n) {
                Some(file::Line::Thought { .. } | file::Line::Heading { level: 2.., .. } | file::Line::Blank | file::Line::Invalid { .. }) => {
                    lines.remove(n);
                    Ok(())
                }
                Some(file::Line::Fact { .. }) => anyhow::bail!("delete facts with the fact's Delete (it's recorded as rejected)"),
                Some(_) => anyhow::bail!("the header line stays"),
                None => anyhow::bail!("no line {n}"),
            }
        })
        .map_err(unprocessable)?;
        Ok(Json(brain_json(&ws, &conn, &model)?))
    })
    .await
}

/// Everything a queue action or fact edit needs, built once per request.
struct Ctx {
    ws: Workspace,
    conn: rusqlite::Connection,
    llm: Ollama,
    embedder: OllamaEmbedder,
    judge: String,
    num_ctx: u32,
}

fn ctx(state: &AppState, model: &str) -> Result<Ctx, ApiError> {
    let mut ws = state.workspace()?;
    let conn = db::open(&ws.db_path())?;
    let llm = Ollama::new(&ws.global.ollama_host)?;
    let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
    let judge = ws.config.agent_corpus_model.clone().unwrap_or_else(|| model.to_string());
    let (num_ctx, _) = budget::resolve_num_ctx(&mut ws, &llm, &judge)?;
    Ok(Ctx { ws, conn, llm, embedder, judge, num_ctx })
}

impl Ctx {
    fn gate<'a>(&'a self, model: &'a str) -> queue::Gate<'a> {
        queue::Gate { ws: &self.ws, conn: &self.conn, embedder: &self.embedder, llm: &self.llm, model, judge_model: &self.judge, num_ctx: self.num_ctx }
    }
}

#[derive(Deserialize)]
struct FactEdit {
    claim: String,
}

async fn edit_fact(State(state): State<AppState>, Path((model, id)): Path<(String, String)>, headers: HeaderMap, Json(body): Json<FactEdit>) -> ApiResult {
    blocking(move || {
        let c = ctx(&state, &model)?;
        c.gate(&model).edit_fact(&id, &body.claim, if_match(&headers).as_deref()).map_err(unprocessable)?;
        Ok(Json(brain_json(&c.ws, &c.conn, &model)?))
    })
    .await
}

async fn delete_fact(State(state): State<AppState>, Path((model, id)): Path<(String, String)>, headers: HeaderMap) -> ApiResult {
    blocking(move || {
        let c = ctx(&state, &model)?;
        c.gate(&model).delete_fact(&id, if_match(&headers).as_deref()).map_err(unprocessable)?;
        Ok(Json(brain_json(&c.ws, &c.conn, &model)?))
    })
    .await
}

async fn restore_fact(State(state): State<AppState>, Path((model, id)): Path<(String, String)>) -> ApiResult {
    blocking(move || {
        let c = ctx(&state, &model)?;
        c.gate(&model).restore_fact(&id).map_err(unprocessable)?;
        Ok(Json(brain_json(&c.ws, &c.conn, &model)?))
    })
    .await
}

async fn organize_brain(State(state): State<AppState>, Path(model): Path<String>) -> ApiResult {
    blocking(move || {
        let mut ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let llm = Ollama::new(&ws.global.ollama_host)?;
        let (num_ctx, _) = budget::resolve_num_ctx(&mut ws, &llm, &model)?;
        let (_, checked) = brain::load_checked(&ws, &conn, &model)?;
        let r = organize::organize(&ws, &llm, &model, num_ctx, &checked).map_err(unprocessable)?;
        Ok(Json(json!({ "result": r, "brain": brain_json(&ws, &conn, &model)? })))
    })
    .await
}

async fn undo(State(state): State<AppState>, Path(model): Path<String>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        file::undo(&ws, &model).map_err(unprocessable)?;
        Ok(Json(brain_json(&ws, &conn, &model)?))
    })
    .await
}

/// The last saved version and the current one, for the diff.
async fn last_change(State(state): State<AppState>, Path(model): Path<String>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let current = file::load(&ws, &model)?.text;
        let before = file::history(&ws, &model).into_iter().next().map(std::fs::read_to_string).transpose().map_err(anyhow::Error::from)?;
        Ok(Json(json!({ "before": before, "after": current })))
    })
    .await
}

async fn queue_list(State(state): State<AppState>, Path(model): Path<String>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let items = queue::pending(&conn, &model)?;
        let mut out = Vec::new();
        for i in items {
            let (cites, status) = brain::integrity::resolve_citations(&conn, &model, None, &i.citations)?;
            let target = match &i.target_fact_id {
                Some(t) => brain::registry::get(&conn, &model, t)?.map(|f| json!({ "id": f.id, "claim": f.claim, "line": f.line })),
                None => None,
            };
            let mut v = serde_json::to_value(&i).map_err(anyhow::Error::from)?;
            v["citations"] = serde_json::to_value(cites).map_err(anyhow::Error::from)?;
            v["sources_ok"] = json!(status == "active");
            v["target"] = target.unwrap_or(Value::Null);
            out.push(v);
        }
        let facts: Vec<Value> = brain::load_checked(&ws, &conn, &model)?.1.lines.iter().filter_map(|l| if let LineView::Fact(f) = l { Some(json!({ "id": f.id, "claim": f.claim })) } else { None }).collect();
        Ok(Json(json!({ "items": out, "facts": facts })))
    })
    .await
}

#[derive(Deserialize, Default)]
struct QueueBody {
    claim: Option<String>,
    target: Option<String>,
}

async fn queue_action(State(state): State<AppState>, Path((model, id, action)): Path<(String, i64, String)>, body: Option<Json<QueueBody>>) -> ApiResult {
    let body = body.map(|b| b.0).unwrap_or_default();
    blocking(move || {
        let c = ctx(&state, &model)?;
        let g = c.gate(&model);
        let fact = match action.as_str() {
            "approve" => Some(g.approve(id, body.claim.as_deref()).map_err(unprocessable)?),
            "reject" => {
                g.reject(id).map_err(unprocessable)?;
                None
            }
            "merge" => {
                let target = body.target.ok_or_else(|| ApiError::new(StatusCode::BAD_REQUEST, "invalid", "name the fact to merge into"))?;
                Some(g.merge(id, &target).map_err(unprocessable)?)
            }
            _ => return Err(not_found("no such queue action")),
        };
        Ok(Json(json!({ "fact_id": fact, "brain": brain_json(&c.ws, &c.conn, &model)? })))
    })
    .await
}

#[derive(Deserialize)]
struct RunsQuery {
    #[serde(default)]
    mode: Mode,
}

async fn runs(State(state): State<AppState>, axum::extract::Query(q): axum::extract::Query<RunsQuery>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        Ok(Json(json!({ "runs": localrag_core::agent::runs(&conn, q.mode, 200)? })))
    })
    .await
}

async fn run_detail(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let (summary, result) = localrag_core::agent::run_detail(&conn, id)?.ok_or_else(|| not_found("no such run"))?;
        Ok(Json(json!({ "run": summary, "result": result })))
    })
    .await
}

async fn delete_run(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        if !localrag_core::agent::delete_run(&conn, id)? {
            return Err(ApiError::new(StatusCode::CONFLICT, "running", "That run is still going (or doesn't exist)."));
        }
        Ok(Json(json!({ "deleted": id })))
    })
    .await
}

#[derive(Deserialize)]
struct RunBody {
    question: String,
    model: String,
    /// research (default) or report.
    #[serde(default)]
    mode: Mode,
}

/// Runs the agent (research, or a report); SSE: `step`, `thinking`, `token`, `tool_call`, `tool_result`,
/// `brain_update`, `warning`, then `result` (or `error`).
async fn run(State(state): State<AppState>, Json(body): Json<RunBody>) -> Result<Response, ApiError> {
    let question = body.question.trim().to_string();
    if question.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "empty", "Type a question first."));
    }
    let root = state.root().ok_or(crate::NoWorkspace).map_err(anyhow::Error::from)?;
    let model = body.model.clone();
    let mode = body.mode;
    {
        let root = root.clone();
        let model = model.clone();
        blocking(move || {
            let ws = Workspace::open(&root)?;
            let info = Ollama::new(&ws.global.ollama_host)?.model_info(&model)?;
            if !(info.can("tools") && info.can("thinking")) {
                return Err(ApiError::new(StatusCode::BAD_REQUEST, "not_reasoning", format!("{model} isn't a reasoning model (it needs Ollama's tools and thinking capabilities).")));
            }
            Ok(())
        })
        .await?;
    }
    let (tx, rx) = tokio::sync::mpsc::channel::<SseEvent>(512);
    tokio::task::spawn_blocking(move || {
        let send = |name: &str, data: Value| {
            let _ = tx.blocking_send(SseEvent::default().event(name).data(data.to_string()));
        };
        let result = (|| -> anyhow::Result<()> {
            let mut ws = Workspace::open(&root)?;
            let conn = db::open(&ws.db_path())?;
            let llm = Ollama::new(&ws.global.ollama_host)?;
            let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
            let corpus_model = ws.config.agent_corpus_model.clone().unwrap_or_else(|| model.clone());
            let (num_ctx, _) = budget::resolve_num_ctx(&mut ws, &llm, &model)?;
            let (corpus_num_ctx, _) = budget::resolve_num_ctx(&mut ws, &llm, &corpus_model)?;
            let agent = Agent { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm, model: &model, num_ctx, corpus_model: &corpus_model, corpus_num_ctx };
            let r = agent.run_mode(mode, &question, &mut |e| match e {
                AgentEvent::Step(n) => send("step", json!({ "step": n })),
                AgentEvent::Thinking(t) => send("thinking", json!({ "text": t })),
                AgentEvent::Token(t) => send("token", json!({ "text": t })),
                AgentEvent::ToolCall { name, arguments } => send("tool_call", json!({ "name": name, "arguments": arguments })),
                AgentEvent::ToolResult { name, result } => send("tool_result", json!({ "name": name, "result": result })),
                AgentEvent::BrainUpdate(v) => send("brain_update", v.clone()),
                AgentEvent::Warning(w) => send("warning", json!({ "message": w })),
            })?;
            send("result", serde_json::to_value(&r)?);
            Ok(())
        })();
        if let Err(e) = result {
            send("error", json!({ "message": format!("{e:#}") }));
        }
    });
    let stream = ReceiverStream::new(rx).map(Ok::<_, Infallible>);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}
