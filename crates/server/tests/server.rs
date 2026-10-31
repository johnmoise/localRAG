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

/// Points the global config (known workspaces, instance files) at a temp dir so tests never
/// touch the real ~/.config/localrag.
fn isolate_config() {
    static DIR: std::sync::OnceLock<tempfile::TempDir> = std::sync::OnceLock::new();
    let dir = DIR.get_or_init(|| tempfile::tempdir().unwrap());
    // SAFETY: every test sets the same value before any config is read.
    unsafe { std::env::set_var("LOCALRAG_CONFIG_DIR", dir.path()) };
}

fn setup() -> (tempfile::TempDir, AppState) {
    isolate_config();
    let dir = tempfile::tempdir().unwrap();
    Workspace::init(dir.path()).unwrap();
    let root = dir.path().canonicalize().unwrap();
    (dir, AppState::new(Some(&root), PORT, TOKEN.into()))
}

fn json_body(v: Value) -> Body {
    Body::from(v.to_string())
}

async fn send_json(state: &AppState, r: Request<Body>) -> (StatusCode, Value) {
    let (status, _, body) = send(state, r).await;
    (status, serde_json::from_slice(&body).unwrap_or(Value::Null))
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
    assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
    assert_eq!(headers[header::X_FRAME_OPTIONS], "DENY");
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
    let ws = Workspace::open(&s.root().unwrap()).unwrap();
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

#[tokio::test]
async fn without_a_workspace_data_routes_ask_for_one() {
    isolate_config();
    let s = AppState::new(None, PORT, TOKEN.into());
    let (status, body) = send_json(&s, authed("GET", "/api/sessions").body(Body::empty()).unwrap()).await;
    assert_eq!((status, body["error"].as_str()), (StatusCode::CONFLICT, Some("no_workspace")));
    let (status, body) = send_json(&s, authed("GET", "/api/status").body(Body::empty()).unwrap()).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["workspace"].is_null());
}

#[tokio::test]
async fn workspaces_can_be_created_opened_and_switched() {
    isolate_config();
    let s = AppState::new(None, PORT, TOKEN.into());
    let parent = tempfile::tempdir().unwrap();
    let post = |path: &str, v: Value| authed("POST", path).header(header::CONTENT_TYPE, "application/json").body(json_body(v)).unwrap();

    let (status, body) = send_json(&s, post("/api/workspaces/create", json!({ "parent": parent.path(), "name": "topic-a" }))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let a = parent.path().canonicalize().unwrap().join("topic-a");
    assert_eq!(s.root().as_deref(), Some(a.as_path()));
    assert!(a.join("corpus").is_dir() && a.join(".localrag/.gitignore").is_file());

    let (status, body) = send_json(&s, post("/api/workspaces/create", json!({ "parent": parent.path(), "name": "topic-a" }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "a non-empty folder isn't overwritten");
    assert!(body["message"].as_str().unwrap().contains("already exists"), "{body}");

    // A plain folder isn't opened silently; with init it becomes a workspace.
    let plain = parent.path().join("plain");
    std::fs::create_dir(&plain).unwrap();
    std::fs::write(plain.join("x.txt"), "x").unwrap();
    let (status, body) = send_json(&s, post("/api/workspaces/open", json!({ "path": plain }))).await;
    assert_eq!((status, body["error"].as_str()), (StatusCode::CONFLICT, Some("not_workspace")));
    let (status, _) = send_json(&s, post("/api/workspaces/open", json!({ "path": plain, "init": true }))).await;
    assert_eq!(status, StatusCode::OK);
    assert!(s.root().unwrap().ends_with("plain"));

    let (_, list) = send_json(&s, authed("GET", "/api/workspaces").body(Body::empty()).unwrap()).await;
    let names: Vec<&str> = list["known"].as_array().unwrap().iter().filter_map(|w| w["name"].as_str()).collect();
    assert!(names.contains(&"topic-a") && names.contains(&"plain"), "{names:?}");
    let (status, _) = send_json(&s, post("/api/workspaces/forget", json!({ "path": s.root().unwrap() }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "the open workspace can't be removed from the list");
}

#[tokio::test]
async fn uploads_are_validated_before_joining_the_corpus() {
    let (_d, s) = setup();
    let up = |name: &str, bytes: &[u8]| {
        authed("POST", &format!("/api/documents/upload?name={}", urlencode(name))).body(Body::from(bytes.to_vec())).unwrap()
    };
    let corpus = s.root().unwrap().join("corpus");

    let (status, body) = send_json(&s, up("notes.md", b"# Notes\nAttention is quadratic.\n")).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["saved_as"], "notes.md");
    let (_, body) = send_json(&s, up("notes.md", b"# Again\nSecond copy.\n")).await;
    assert_eq!((body["saved_as"].as_str(), body["renamed"].as_bool()), (Some("notes (2).md"), Some(true)));
    let (_, body) = send_json(&s, up("../../escape.txt", b"plain text")).await;
    assert_eq!(body["saved_as"], "escape.txt");
    assert!(corpus.join("escape.txt").is_file(), "path components are stripped");

    let (status, body) = send_json(&s, up("chapter.docx", b"PK\x03\x04")).await;
    assert_eq!(status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    assert!(body["message"].as_str().unwrap().contains("PDF (.pdf), Markdown"));
    let (status, body) = send_json(&s, up("bad.txt", &[b'h', b'i', 0xff])).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["message"].as_str().unwrap().contains("byte 0xff at offset 2"), "{body}");
    let (status, body) = send_json(&s, up("fake.pdf", b"not really a pdf")).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["message"].as_str().unwrap().contains("no PDF header"), "{body}");

    let mut files: Vec<String> = std::fs::read_dir(&corpus).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().to_string()).collect();
    files.sort();
    assert_eq!(files, ["escape.txt", "notes (2).md", "notes.md"], "rejected files never reach corpus/, no temp files left");
}

fn urlencode(s: &str) -> String {
    s.bytes().map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
}
