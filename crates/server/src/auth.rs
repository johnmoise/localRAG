//! Request guard (ARCHITECTURE.md §14). Binding to 127.0.0.1 isn't enough: any web page in
//! the browser can send requests to localhost, and DNS rebinding can get around same-origin
//! rules. So every request must name our host, carry no foreign Origin, and (for data
//! routes) present the per-launch token.

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::middleware::Next;
use axum::response::{IntoResponse, Redirect, Response};

use crate::AppState;

/// Cookies are shared across ports on one host, so each server (workspace) gets its own
/// cookie name: `localrag_token_<port>`. Two workspaces served at once don't log each other out.
pub const TOKEN_COOKIE: &str = "localrag_token";

pub fn cookie_name(port: u16) -> String {
    format!("{TOKEN_COOKIE}_{port}")
}

pub async fn guard(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let host_ok = req
        .headers()
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .is_some_and(|h| h == format!("127.0.0.1:{}", state.port) || h == format!("localhost:{}", state.port));
    if !host_ok {
        return (StatusCode::MISDIRECTED_REQUEST, "unexpected Host header").into_response();
    }
    if let Some(origin) = req.headers().get(header::ORIGIN).and_then(|o| o.to_str().ok())
        && origin != format!("http://127.0.0.1:{}", state.port)
        && origin != format!("http://localhost:{}", state.port)
    {
        return (StatusCode::FORBIDDEN, "cross-origin request refused").into_response();
    }

    let path = req.uri().path();
    // The launch URL `/?token=…` trades the token for a cookie, then drops it from the URL.
    if path == "/" {
        if let Some(t) = req.uri().query().and_then(|q| q.split('&').find_map(|kv| kv.strip_prefix("token=")))
        {
            if constant_time_eq(t, &state.token) {
                let mut resp = Redirect::to("/").into_response();
                let cookie = format!("{}={}; HttpOnly; SameSite=Strict; Path=/", cookie_name(state.port), state.token);
                resp.headers_mut().insert(header::SET_COOKIE, cookie.parse().expect("valid cookie"));
                return resp;
            }
            return (StatusCode::UNAUTHORIZED, "invalid token").into_response();
        }
        if !has_token(req.headers(), state.port, &state.token) {
            return (StatusCode::UNAUTHORIZED, axum::response::Html(LOCKED_PAGE)).into_response();
        }
    }
    let protected = path.starts_with("/api/") || path.starts_with("/files/");
    if protected && !has_token(req.headers(), state.port, &state.token) {
        return (StatusCode::UNAUTHORIZED, "missing or invalid token").into_response();
    }
    next.run(req).await
}

fn has_token(headers: &HeaderMap, port: u16, token: &str) -> bool {
    let name = cookie_name(port);
    headers
        .get_all(header::COOKIE)
        .iter()
        .filter_map(|v| v.to_str().ok())
        .flat_map(|v| v.split(';'))
        .filter_map(|kv| kv.trim().strip_prefix(name.as_str()).and_then(|r| r.strip_prefix('=')))
        .any(|t| constant_time_eq(t, token))
}

fn constant_time_eq(a: &str, b: &str) -> bool {
    a.len() == b.len() && a.bytes().zip(b.bytes()).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

const LOCKED_PAGE: &str = "<!doctype html><meta charset=utf-8><title>localrag</title>\
<body style=\"font:16px system-ui;margin:3rem\"><h1>localrag</h1>\
<p>Open the link printed by <code>localrag serve</code> in your terminal. It carries this session's access token.</p></body>";
