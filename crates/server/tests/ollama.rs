//! Ollama setup status (ARCHITECTURE.md §19.1). Its own test binary: it points the global
//! config's `ollama_host` at a fake server, which must not leak into the other tests.

use std::time::{Duration, Instant};

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use localrag_core::ollama::SearchEnv;
use localrag_server::{AppState, router};
use serde_json::Value;
use tower::ServiceExt;

const PORT: u16 = 4712;
const TOKEN: &str = "test-token-ollama-0123";

async fn call(state: &AppState, method: &str, path: &str) -> (StatusCode, Value) {
    let r = Request::builder()
        .method(method)
        .uri(path)
        .header(header::HOST, format!("127.0.0.1:{PORT}"))
        .header(header::COOKIE, format!("localrag_token_{PORT}={TOKEN}"))
        .body(Body::empty())
        .unwrap();
    let resp = router(state.clone()).oneshot(r).await.unwrap();
    let status = resp.status();
    let body = resp.into_body().collect().await.unwrap().to_bytes();
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
}

/// An Ollama URL that accepts connections and never answers is "not installed" (nothing to
/// start, since the search finds nothing), reported promptly instead of hanging or crashing.
#[tokio::test(flavor = "multi_thread")]
async fn silent_ollama_reports_not_installed() {
    let dir = tempfile::tempdir().unwrap();
    // SAFETY: the only test in this binary; set before any config is read.
    unsafe { std::env::set_var("LOCALRAG_CONFIG_DIR", dir.path()) };
    // Kept open but never accepted from: connects succeed (backlog), requests get no reply.
    let silent = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let host = format!("http://{}", silent.local_addr().unwrap());
    std::fs::write(dir.path().join("config.toml"), format!("ollama_host = \"{host}\"\n")).unwrap();

    let state = AppState::new(None, PORT, TOKEN.into()).with_ollama_search(SearchEnv::none());

    let (status, body) = call(&state, "POST", "/api/ollama/check").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let started = Instant::now();
    let body = loop {
        let (status, body) = call(&state, "GET", "/api/ollama").await;
        assert_eq!(status, StatusCode::OK, "{body}");
        if body["phase"] == "done" {
            break body;
        }
        assert!(started.elapsed() < Duration::from_secs(10), "check never finished: {body}");
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    assert_eq!(body["status"]["state"], "not_installed", "{body}");
    assert_eq!(body["host"], host.as_str());
    assert_eq!(body["local"], true);
    assert_eq!(body["download_url"], "https://ollama.com/download");
    assert_eq!(body["models"].as_array().map(Vec::len), Some(0), "no model list without Ollama");

    // Pulling with Ollama silent fails cleanly too.
    let (status, body) = call(&state, "POST", "/api/ollama/pull").await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert!(body["pull"]["error"].is_string(), "{body}");
    assert_eq!(body["pull"]["running"], false);
    drop(silent);
}
