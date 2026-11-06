//! HTTP routes (ARCHITECTURE.md §14).

use std::convert::Infallible;

use axum::extract::{Path, Query, State};
use axum::http::{StatusCode, header};
use axum::response::sse::{Event as SseEvent, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use rust_embed::RustEmbed;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio_stream::wrappers::ReceiverStream;
use tokio_stream::StreamExt;

use localrag_core::answer::{Event, RagEngine};
use localrag_core::budget::{self, OverBudget};
use localrag_core::embed::OllamaEmbedder;
use localrag_core::llm::Ollama;
use localrag_core::retrieval::{self, SearchOptions};
use localrag_core::{cite, citations, db, highlight, history, index, refs};

use crate::AppState;

#[derive(RustEmbed)]
#[folder = "../../web/"]
struct Assets;

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index_html))
        .route("/assets/{*path}", get(asset))
        .route("/api/status", get(status))
        .route("/api/workspaces", get(crate::workspaces::list))
        .route("/api/workspaces/open", post(crate::workspaces::open))
        .route("/api/workspaces/create", post(crate::workspaces::create))
        .route("/api/workspaces/forget", post(crate::workspaces::forget))
        .route("/api/workspaces/choose-folder", post(crate::workspaces::choose_folder))
        .route(
            "/api/documents/upload",
            post(crate::upload::upload).layer(axum::extract::DefaultBodyLimit::max(crate::upload::MAX_UPLOAD_BYTES)),
        )
        .route("/api/ollama", get(crate::ollama::get))
        .route("/api/ollama/check", post(crate::ollama::check))
        .route("/api/ollama/pull", post(crate::ollama::pull))
        .route("/api/models", get(models))
        .route("/api/search", post(search))
        .route("/api/ask", post(ask))
        .route("/api/sessions", get(sessions))
        .route("/api/sessions/{id}", get(session).delete(delete_session))
        .route("/api/messages/{id}/citations/{ordinal}/paraphrase", post(paraphrase))
        .route("/api/documents", get(documents))
        .route("/api/documents/{id}", delete(forget))
        .route("/api/documents/forget-removed", post(forget_removed))
        .route("/api/documents/{id}/rename", post(rename))
        .route("/api/documents/{id}/metadata", get(metadata_get).patch(metadata_edit))
        .route("/api/documents/{id}/metadata/refresh", post(metadata_refresh))
        .route("/api/documents/{id}/metadata/resolve", post(metadata_resolve))
        .route("/api/documents/{id}/ocr", post(ocr_one))
        .route("/api/documents/{id}/references", get(references))
        .route("/api/ocr/all", post(ocr_all))
        .route("/api/ocr/cancel", post(ocr_cancel))
        .route("/api/settings", get(crate::settings::get).put(crate::settings::put))
        .route("/api/reindex", post(reindex))
        .route("/api/chunks/{id}", get(chunk))
        .route("/api/highlights", get(highlights))
        .route("/files/{id}", get(file))
        .merge(crate::brain::routes())
        .with_state(state)
}

/// A JSON error: `{"error": kind, "message": …, …extra}`.
pub struct ApiError {
    status: StatusCode,
    body: Value,
}

impl ApiError {
    pub fn new(status: StatusCode, kind: &str, message: impl Into<String>) -> Self {
        ApiError { status, body: json!({ "error": kind, "message": message.into() }) }
    }
}

impl From<anyhow::Error> for ApiError {
    fn from(e: anyhow::Error) -> Self {
        if e.downcast_ref::<crate::NoWorkspace>().is_some() {
            return ApiError::new(StatusCode::CONFLICT, "no_workspace", e.to_string());
        }
        if e.downcast_ref::<localrag_core::brain::file::Conflict>().is_some() {
            return ApiError::new(StatusCode::CONFLICT, "brain_changed", e.to_string());
        }
        if let Some(over) = e.downcast_ref::<OverBudget>() {
            let mut body = serde_json::to_value(over).unwrap_or_default();
            body["error"] = json!("over_budget");
            body["message"] = json!(over.to_string());
            return ApiError { status: StatusCode::PAYLOAD_TOO_LARGE, body };
        }
        ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", format!("{e:#}"))
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body)).into_response()
    }
}

pub type ApiResult = Result<Json<Value>, ApiError>;

/// Runs blocking core work off the async runtime.
pub async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, ApiError> + Send + 'static) -> Result<T, ApiError> {
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| ApiError::new(StatusCode::INTERNAL_SERVER_ERROR, "internal", e.to_string()))?
}

pub(crate) fn not_found(what: &str) -> ApiError {
    ApiError::new(StatusCode::NOT_FOUND, "not_found", what)
}

/// Scripts and styles only from this server; PDF.js needs blob:/data: for its worker and fonts.
const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; \
    font-src 'self' data:; worker-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'";

async fn index_html() -> Response {
    let mut resp = asset_response("index.html");
    resp.headers_mut().insert(header::CONTENT_SECURITY_POLICY, CSP.parse().expect("valid CSP"));
    resp
}

async fn asset(Path(path): Path<String>) -> Response {
    asset_response(&path)
}

fn asset_response(path: &str) -> Response {
    match Assets::get(path) {
        Some(file) => {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            ([(header::CONTENT_TYPE, mime.as_ref().to_string()), (header::CACHE_CONTROL, "no-cache".to_string())], file.data)
                .into_response()
        }
        None => (StatusCode::NOT_FOUND, "not found").into_response(),
    }
}

async fn status(State(state): State<AppState>) -> ApiResult {
    let sync = state.sync.lock().expect("sync state").clone();
    let Some(root) = state.root() else {
        return Ok(Json(json!({ "workspace": null })));
    };
    blocking(move || {
        let ws = localrag_core::workspace::Workspace::open(&root)?;
        let conn = db::open(&ws.db_path())?;
        let s = index::status(&conn)?;
        let ollama = Ollama::new(&ws.global.ollama_host).and_then(|o| o.version());
        Ok(Json(json!({
            "workspace": ws.root,
            "name": ws.root.file_name().map(|n| n.to_string_lossy().to_string()),
            "documents": s.by_status.iter().map(|(k, n)| (k.clone(), json!(n))).collect::<serde_json::Map<_, _>>(),
            "chunks": s.chunks,
            "embed_model": s.embed_model,
            "last_sync": s.last_sync,
            "sync": sync,
            "ollama": match ollama {
                Ok(v) => json!({ "reachable": true, "version": v, "host": ws.global.ollama_host }),
                Err(e) => json!({ "reachable": false, "error": format!("{e:#}"), "host": ws.global.ollama_host }),
            },
        })))
    })
    .await
}

async fn models(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let models = Ollama::new(&ws.global.ollama_host)?.chat_models()?;
        Ok(Json(json!({
            "models": models.iter().map(|m| json!({
                "name": m.name, "family": m.family, "context_length": m.context_length, "capabilities": m.capabilities,
            })).collect::<Vec<_>>(),
            "default": ws.config.chat_model,
            "num_ctx": ws.config.num_ctx,
        })))
    })
    .await
}

#[derive(Deserialize)]
struct SearchBody {
    query: String,
    #[serde(default = "ten")]
    k: usize,
    #[serde(default)]
    filter: retrieval::Filter,
}

fn ten() -> usize {
    10
}

async fn search(State(state): State<AppState>, Json(body): Json<SearchBody>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
        let opts = SearchOptions { candidates: ws.config.retrieval.candidates, limit: body.k.clamp(1, 50), per_doc_cap: ws.config.retrieval.per_doc_cap, filter: body.filter };
        let hits = retrieval::search(&conn, &embedder, &body.query, &opts)?;
        Ok(Json(json!({ "hits": hits })))
    })
    .await
}

#[derive(Deserialize)]
struct AskBody {
    question: String,
    model: Option<String>,
    session_id: Option<i64>,
}

/// Budget check first (413 "Over token count by X"), then an SSE stream of the answer.
async fn ask(State(state): State<AppState>, Json(body): Json<AskBody>) -> Result<Response, ApiError> {
    let question = body.question.trim().to_string();
    if question.is_empty() {
        return Err(ApiError::new(StatusCode::BAD_REQUEST, "empty", "Type a question first."));
    }
    let root = state.root().ok_or(crate::NoWorkspace).map_err(anyhow::Error::from)?;
    let (model, num_ctx) = {
        let question = question.clone();
        let root = root.clone();
        let model = body.model.clone();
        blocking(move || {
            let mut ws = localrag_core::workspace::Workspace::open(&root)?;
            let conn = db::open(&ws.db_path())?;
            let llm = Ollama::new(&ws.global.ollama_host)?;
            let model = model.unwrap_or_else(|| ws.config.chat_model.clone());
            let (num_ctx, _) = budget::resolve_num_ctx(&mut ws, &llm, &model)?;
            let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
            RagEngine { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm }.check_budget(&question, &model, num_ctx)?;
            Ok((model, num_ctx))
        })
        .await?
    };

    let (tx, rx) = tokio::sync::mpsc::channel::<SseEvent>(256);
    tokio::task::spawn_blocking(move || {
        let send = |name: &str, data: Value| {
            let _ = tx.blocking_send(SseEvent::default().event(name).data(data.to_string()));
        };
        let result = (|| -> anyhow::Result<()> {
            let ws = localrag_core::workspace::Workspace::open(&root)?;
            let conn = db::open(&ws.db_path())?;
            let llm = Ollama::new(&ws.global.ollama_host)?;
            let embedder = OllamaEmbedder::new(&ws.global.ollama_host, &ws.config.embed_model)?;
            let session = match body.session_id {
                Some(id) if history::session_exists(&conn, id)? => Some(id),
                _ => None,
            };
            let turns = match session {
                Some(id) => history::turns(&conn, id)?,
                None => Vec::new(),
            };
            let engine = RagEngine { ws: &ws, conn: &conn, embedder: &embedder, llm: &llm };
            let answer = engine.ask(&question, &model, num_ctx, &turns, &mut |e| match e {
                Event::Query(q) => send("query", json!({ "query": q })),
                Event::Sources(s) => send("sources", json!({ "sources": s })),
                Event::Warning(w) => send("warning", json!({ "warning": w, "message": w.to_string() })),
                Event::Thinking(t) => send("thinking", json!({ "text": t })),
                Event::Token(t) => send("token", json!({ "text": t })),
            })?;
            let session = match session {
                Some(id) => id,
                None => history::create_session(&conn, &question, &model)?,
            };
            let message_id = history::record(&conn, session, &question, &answer)?;
            send("answer", json!({
                "answer": answer,
                "placeholder_markdown": citations::placeholders(&answer.text),
                "session_id": session,
                "message_id": message_id,
            }));
            Ok(())
        })();
        if let Err(e) = result {
            send("error", json!({ "message": format!("{e:#}") }));
        }
    });
    let stream = ReceiverStream::new(rx).map(Ok::<_, Infallible>);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()).into_response())
}

async fn sessions(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        Ok(Json(json!({ "sessions": history::sessions(&conn)? })))
    })
    .await
}

async fn session(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        if !history::session_exists(&conn, id)? {
            return Err(not_found("no such session"));
        }
        Ok(Json(json!({ "id": id, "messages": history::messages(&conn, id)? })))
    })
    .await
}

async fn delete_session(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        history::delete_session(&db::open(&ws.db_path())?, id)?;
        Ok(Json(json!({ "deleted": id })))
    })
    .await
}

#[derive(Deserialize)]
struct ParaphraseBody {
    on: bool,
}

async fn paraphrase(State(state): State<AppState>, Path((id, ordinal)): Path<(i64, usize)>, Json(body): Json<ParaphraseBody>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let msg = history::set_paraphrase(&conn, id, ordinal, body.on)
            .map_err(|e| ApiError::new(StatusCode::CONFLICT, "conflict", format!("{e:#}")))?;
        Ok(Json(json!({ "message": msg })))
    })
    .await
}

async fn documents(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let mut stmt = conn.prepare(
            "SELECT d.id, d.path, d.status, d.kind, (SELECT count(*) FROM chunks c WHERE c.doc_id = d.id),
                    d.error, d.scanned_pages, d.cite_label, d.csl_json, d.metadata_source, d.indexed_at, d.removed_at, d.page_count,
                    d.metadata_status, d.doi, d.reference_count, d.cited_by,
                    (SELECT count(*) FROM doc_references r WHERE r.doc_id = d.id AND r.in_corpus_doc_id IS NOT NULL)
             FROM documents d ORDER BY d.path",
        ).map_err(anyhow::Error::from)?;
        let docs: Vec<Value> = stmt
            .query_map([], |r| {
                let path: String = r.get(1)?;
                let cite_label: Option<String> = r.get(7)?;
                let csl: Option<String> = r.get(8)?;
                let scanned: Option<String> = r.get(6)?;
                Ok(json!({
                    "id": r.get::<_, i64>(0)?,
                    "label": cite::doc_label(cite_label.as_deref(), csl.as_deref(), &path),
                    "path": path,
                    "status": r.get::<_, String>(2)?,
                    "kind": r.get::<_, String>(3)?,
                    "chunks": r.get::<_, i64>(4)?,
                    "error": r.get::<_, Option<String>>(5)?,
                    "scanned_pages": scanned.and_then(|s| serde_json::from_str::<Value>(&s).ok()),
                    "metadata_source": r.get::<_, Option<String>>(9)?,
                    "indexed_at": r.get::<_, Option<String>>(10)?,
                    "removed_at": r.get::<_, Option<String>>(11)?,
                    "pages": r.get::<_, Option<i64>>(12)?,
                    "metadata_status": r.get::<_, Option<String>>(13)?,
                    "doi": r.get::<_, Option<String>>(14)?,
                    "reference_count": r.get::<_, Option<i64>>(15)?,
                    "cited_by": r.get::<_, Option<i64>>(16)?,
                    "references_in_corpus": r.get::<_, i64>(17)?,
                    "csl": csl.as_deref().and_then(|c| serde_json::from_str::<Value>(c).ok()),
                }))
            })
            .map_err(anyhow::Error::from)?
            .collect::<Result<_, _>>()
            .map_err(anyhow::Error::from)?;
        Ok(Json(json!({ "documents": docs })))
    })
    .await
}

async fn forget(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let path = index::forget(&ws, &conn, id).map_err(|e| ApiError::new(StatusCode::CONFLICT, "conflict", format!("{e:#}")))?;
        Ok(Json(json!({ "forgotten": path })))
    })
    .await
}

async fn forget_removed(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let mut forgotten = Vec::new();
        for d in index::list_documents(&conn)?.into_iter().filter(|d| d.status == "removed") {
            forgotten.push(index::forget(&ws, &conn, d.id)?);
        }
        Ok(Json(json!({ "forgotten": forgotten })))
    })
    .await
}

async fn metadata_get(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        Ok(Json(localrag_core::metadata::summary(&db::open(&ws.db_path())?, id)?))
    })
    .await
}

async fn metadata_edit(State(state): State<AppState>, Path(id): Path<i64>, Json(m): Json<localrag_core::metadata::ManualMetadata>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        localrag_core::metadata::set_manual(&conn, id, &m)?;
        Ok(Json(localrag_core::metadata::summary(&conn, id)?))
    })
    .await
}

async fn metadata_refresh(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let client = localrag_core::metadata::crossref::configured(&ws.global);
        let event = localrag_core::metadata::refresh(&conn, client, id)
            .map_err(|e| ApiError::new(StatusCode::BAD_GATEWAY, "crossref", format!("{e:#}")))?;
        Ok(Json(json!({ "event": event, "metadata": localrag_core::metadata::summary(&conn, id)? })))
    })
    .await
}

#[derive(Deserialize)]
struct ResolveBody {
    /// Index of the accepted candidate, or null to reject them all.
    accept: Option<usize>,
}

async fn metadata_resolve(State(state): State<AppState>, Path(id): Path<i64>, Json(b): Json<ResolveBody>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        localrag_core::metadata::resolve_proposal(&conn, id, b.accept)?;
        Ok(Json(localrag_core::metadata::summary(&conn, id)?))
    })
    .await
}

async fn references(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        Ok(Json(json!({ "references": localrag_core::references::for_document(&conn, id)? })))
    })
    .await
}

async fn ocr_one(State(state): State<AppState>, Path(id): Path<i64>) -> ApiResult {
    if !state.start_ocr(vec![id]) {
        return Err(ApiError::new(StatusCode::CONFLICT, "busy", "Another job is running; try again when it finishes."));
    }
    Ok(Json(json!({ "started": true })))
}

async fn ocr_all(State(state): State<AppState>) -> ApiResult {
    let ids = {
        let ws = state.workspace()?;
        index::needing_ocr(&db::open(&ws.db_path())?)?.into_iter().map(|d| d.0).collect::<Vec<_>>()
    };
    if !state.start_ocr(ids) {
        return Err(ApiError::new(StatusCode::CONFLICT, "busy", "Another job is running; try again when it finishes."));
    }
    Ok(Json(json!({ "started": true })))
}

async fn ocr_cancel() -> ApiResult {
    crate::ocr_jobs::CANCEL.store(true, std::sync::atomic::Ordering::SeqCst);
    Ok(Json(json!({ "cancelling": true })))
}

#[derive(Deserialize)]
struct RenameBody {
    name: String,
}

/// Renames a corpus file without reindexing (§15).
async fn rename(State(state): State<AppState>, Path(id): Path<i64>, Json(body): Json<RenameBody>) -> ApiResult {
    if state.sync.lock().expect("sync state").running {
        return Err(ApiError::new(StatusCode::CONFLICT, "busy", "Indexing is running; try renaming again when it finishes."));
    }
    blocking(move || {
        let ws = state.workspace()?;
        let mut conn = db::open(&ws.db_path())?;
        let name = crate::upload::safe_name(&body.name)?;
        let report = index::rename_document(&ws, &mut conn, id, &name)
            .map_err(|e| ApiError::new(StatusCode::CONFLICT, "rename_failed", format!("{e:#}")))?;
        Ok(Json(serde_json::to_value(report).map_err(anyhow::Error::from)?))
    })
    .await
}

async fn reindex(State(state): State<AppState>) -> ApiResult {
    let started = state.start_sync();
    Ok(Json(json!({ "started": started })))
}

#[derive(Deserialize)]
struct ChunkQuery {
    /// From a chat-history snapshot: resolve by content when the id no longer matches.
    text_sha: Option<String>,
    doc_sha: Option<String>,
    /// A verified quote to locate in the chunk (snapshots may not carry its span).
    quote: Option<String>,
}

async fn chunk(State(state): State<AppState>, Path(id): Path<i64>, Query(q): Query<ChunkQuery>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        // A history snapshot's chunk id can go stale after reindexing; then resolve by content.
        let direct = retrieval::load_chunk(&conn, id)?.filter(|c| q.text_sha.as_ref().is_none_or(|sha| &c.text_sha == sha));
        let (chunk, resolution) = match (direct, &q.text_sha) {
            (Some(c), _) => {
                let removed = conn.query_row("SELECT status = 'removed' FROM documents WHERE id = ?1", [c.doc_id], |r| r.get(0)).map_err(anyhow::Error::from)?;
                (Some(c), if removed { refs::Resolution::SourceRemoved } else { refs::Resolution::Resolved })
            }
            (None, Some(sha)) => {
                let r = refs::resolve(&conn, q.doc_sha.as_deref().unwrap_or(""), sha, None)?;
                let c = match r.chunk_id {
                    Some(id) => retrieval::load_chunk(&conn, id)?,
                    None => None,
                };
                (c, r.resolution)
            }
            (None, None) => (None, refs::Resolution::Broken),
        };
        let chunk = chunk.ok_or_else(|| not_found("that passage is no longer in the index (forgotten or changed)"))?;
        let span = q.quote.as_deref().and_then(|qt| citations::verify_quote(qt, &chunk.text, chunk.ocr)).map(|r| [r.start, r.end]);
        Ok(Json(json!({ "chunk": chunk, "resolution": resolution, "span": span })))
    })
    .await
}

#[derive(Deserialize)]
struct HighlightQuery {
    chunk: i64,
    start: Option<usize>,
    end: Option<usize>,
}

async fn highlights(State(state): State<AppState>, Query(q): Query<HighlightQuery>) -> ApiResult {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let span = q.start.zip(q.end);
        Ok(Json(serde_json::to_value(highlight::for_chunk(&ws, &conn, q.chunk, span)?).map_err(anyhow::Error::from)?))
    })
    .await
}

/// The original file, looked up by document id only (never by a client-supplied path).
async fn file(State(state): State<AppState>, Path(id): Path<i64>) -> Result<Response, ApiError> {
    blocking(move || {
        let ws = state.workspace()?;
        let conn = db::open(&ws.db_path())?;
        let (path, kind): (String, String) = conn
            .query_row("SELECT path, kind FROM documents WHERE id = ?1 AND status != 'removed'", [id], |r| Ok((r.get(0)?, r.get(1)?)))
            .map_err(|_| not_found("no such document in corpus/"))?;
        let bytes = std::fs::read(ws.corpus_dir().join(&path)).map_err(|_| not_found("file is missing from corpus/"))?;
        let mime = if kind == "pdf" { "application/pdf" } else { "text/plain; charset=utf-8" };
        Ok(([(header::CONTENT_TYPE, mime), (header::CONTENT_DISPOSITION, "inline")], bytes).into_response())
    })
    .await
}
