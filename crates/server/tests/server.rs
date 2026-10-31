//! Request guard and API behaviour (ARCHITECTURE.md §14).

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use http_body_util::BodyExt;
use localrag_core::db;
use localrag_core::workspace::Workspace;
use localrag_server::{AppState, router};
use serde_json::{Value, json};
use tower::ServiceExt;

const PORT: u16 = 4711;
const TOKEN: &str = "test-token-0123456789";

fn setup() -> (tempfile::TempDir, AppState) {
    let dir = tempfile::tempdir().unwrap();
    Workspace::init(dir.path()).unwrap();
    let root = dir.path().canonicalize().unwrap();
    (dir, AppState::new(&root, PORT, TOKEN.into()))
}

fn req(method: &str, path: &str) -> axum::http::request::Builder {
    Request::builder().method(method).uri(path).header(header::HOST, format!("127.0.0.1:{PORT}"))
}

fn authed(method: &str, path: &str) -> axum::http::request::Builder {
    req(method, path).header(header::COOKIE, format!("other=1; localrag_token_{PORT}={TOKEN}"))
}

async fn send(state: &AppState, r: Request<Body>) -> (StatusCode, axum::http::HeaderMap, Vec<u8>) {
    let resp = router(state.clone()).oneshot(r).await.unwrap();
    let status = resp.status();
    let headers = resp.headers().clone();
    let body = resp.into_body().collect().await.unwrap().to_bytes().to_vec();
    (status, headers, body)
}

#[tokio::test]
async fn rejects_foreign_hosts_and_origins() {
    let (_d, s) = setup();
    let r = Request::builder().uri("/api/status").header(header::HOST, "evil.example:4711").body(Body::empty()).unwrap();
    assert_eq!(send(&s, r).await.0, StatusCode::MISDIRECTED_REQUEST, "DNS rebinding: wrong Host");
    let r = authed("POST", "/api/reindex").header(header::ORIGIN, "https://evil.example").body(Body::empty()).unwrap();
    assert_eq!(send(&s, r).await.0, StatusCode::FORBIDDEN, "CSRF: foreign Origin");
    let r = authed("GET", "/api/sessions").header(header::ORIGIN, format!("http://localhost:{PORT}")).body(Body::empty()).unwrap();
    assert_eq!(send(&s, r).await.0, StatusCode::OK, "own origin is fine");
}

#[tokio::test]
async fn data_routes_need_the_token() {
    let (_d, s) = setup();
    for path in ["/api/sessions", "/api/documents", "/files/1"] {
        assert_eq!(send(&s, req("GET", path).body(Body::empty()).unwrap()).await.0, StatusCode::UNAUTHORIZED, "{path}");
    }
    let wrong_port_cookie = req("GET", "/api/sessions").header(header::COOKIE, format!("localrag_token_9999={TOKEN}"));
    assert_eq!(send(&s, wrong_port_cookie.body(Body::empty()).unwrap()).await.0, StatusCode::UNAUTHORIZED);
    // Static assets carry no data and load without it.
    assert_eq!(send(&s, req("GET", "/assets/app.js").body(Body::empty()).unwrap()).await.0, StatusCode::OK);
    assert_eq!(send(&s, authed("GET", "/api/sessions").body(Body::empty()).unwrap()).await.0, StatusCode::OK);
}

#[tokio::test]
async fn launch_url_trades_token_for_a_per_port_cookie() {
    let (_d, s) = setup();
    let (status, _, body) = send(&s, req("GET", "/").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert!(String::from_utf8_lossy(&body).contains("Open the link printed by"));

    assert_eq!(send(&s, req("GET", "/?token=nope").body(Body::empty()).unwrap()).await.0, StatusCode::UNAUTHORIZED);

    let (status, headers, _) = send(&s, req("GET", &format!("/?token={TOKEN}")).body(Body::empty()).unwrap()).await;
    assert!(status.is_redirection());
    let cookie = headers[header::SET_COOKIE].to_str().unwrap();
    assert!(cookie.starts_with(&format!("localrag_token_{PORT}={TOKEN}")), "{cookie}");
    assert!(cookie.contains("HttpOnly") && cookie.contains("SameSite=Strict"));

    let (status, headers, body) = send(&s, authed("GET", "/").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(String::from_utf8_lossy(&body).contains("/assets/app.js"));
    assert!(headers[header::CONTENT_SECURITY_POLICY].to_str().unwrap().contains("script-src 'self'"));
}

#[tokio::test]
async fn empty_questions_are_rejected() {
    let (_d, s) = setup();
    let r = authed("POST", "/api/ask").header(header::CONTENT_TYPE, "application/json").body(Body::from(r#"{"question":"  "}"#)).unwrap();
    let (status, _, body) = send(&s, r).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(serde_json::from_slice::<Value>(&body).unwrap()["error"], "empty");
}

#[tokio::test]
async fn citations_can_be_promoted_to_verified_paraphrase_and_back() {
    let (_d, s) = setup();
    let ws = Workspace::open(&s.root).unwrap();
    let conn = db::open(&ws.db_path()).unwrap();
    let session = localrag_core::history::create_session(&conn, "q", "m").unwrap();
    let citations = json!([
        {"ordinal": 0, "marker": 1, "label": "A 2020, p. 1", "doc_path": "a.pdf", "doc_sha256": "d", "chunk_text_sha": "c",
         "chunk_id": 1, "quote": "not verbatim", "quote_verified": false, "status": "unverified"},
        {"ordinal": 1, "marker": 2, "label": "B 2021, p. 2", "doc_path": "b.pdf", "doc_sha256": "e", "chunk_text_sha": "f",
         "chunk_id": 2, "quote": "exact", "quote_verified": true, "status": "verified"}
    ]);
    conn.execute(
        "INSERT INTO messages (session_id, role, content, citations_json, supported, created_at) VALUES (?1, 'assistant', ?2, ?3, 0, 'now')",
        rusqlite::params![session, r#"Claim [S1 "not verbatim"]. Other [S2 "exact"]."#, citations.to_string()],
    )
    .unwrap();
    let msg = conn.last_insert_rowid();

    let promote = |ordinal: usize, on: bool| {
        authed("POST", &format!("/api/messages/{msg}/citations/{ordinal}/paraphrase"))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "on": on }).to_string()))
            .unwrap()
    };
    let (status, _, body) = send(&s, promote(0, true)).await;
    assert_eq!(status, StatusCode::OK);
    let m: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(m["message"]["citations"][0]["status"], "verified_paraphrase");
    assert!(m["message"]["citations"][0]["paraphrase_at"].is_string());
    assert_eq!(m["message"]["supported"], true);
    assert_eq!(m["message"]["placeholder_markdown"], "Claim ⟦0⟧. Other ⟦1⟧.");

    assert_eq!(send(&s, promote(1, true)).await.0, StatusCode::CONFLICT, "verified quotes can't be 'promoted'");

    let (_, _, body) = send(&s, promote(0, false)).await;
    let m: Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(m["message"]["citations"][0]["status"], "unverified");
    assert_eq!(m["message"]["supported"], true, "the other citation is still verified");
}
