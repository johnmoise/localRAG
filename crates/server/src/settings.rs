//! Settings (ARCHITECTURE.md §7, §15): the Crossref email lives in the machine-wide config
//! (never inside a workspace); default model, context sizes and OCR mode are per workspace.

use std::collections::BTreeMap;

use axum::Json;
use axum::extract::State;
use axum::http::StatusCode;
use serde::Deserialize;
use serde_json::json;

use localrag_core::config::{self, OcrMode};

use crate::AppState;
use crate::routes::{ApiError, ApiResult, blocking};

pub async fn get(State(state): State<AppState>) -> ApiResult {
    blocking(move || {
        let g = config::load_global()?;
        let ws = state.workspace().ok();
        Ok(Json(json!({
            "crossref_email": g.crossref_email,
            "crossref_email_prompted": g.crossref_email_prompted,
            "pool": if g.crossref_email.is_some() { "polite" } else { "public" },
            "workspace": ws.as_ref().map(|w| json!({
                "chat_model": w.config.chat_model,
                "num_ctx": w.config.num_ctx,
                "ocr": w.config.ocr,
                "think": w.config.think,
                "repair_quotes": w.config.repair_quotes,
                "strict": w.config.strict,
            })),
        })))
    })
    .await
}

#[derive(Deserialize)]
pub struct Update {
    /// `""` or `null` clears it.
    #[serde(default, deserialize_with = "some_or_null")]
    crossref_email: Option<Option<String>>,
    crossref_email_prompted: Option<bool>,
    chat_model: Option<String>,
    num_ctx: Option<BTreeMap<String, u32>>,
    ocr: Option<OcrMode>,
    think: Option<bool>,
    repair_quotes: Option<bool>,
    strict: Option<bool>,
}

/// Distinguishes a missing field (leave as is) from `null` (clear).
fn some_or_null<'de, D: serde::Deserializer<'de>>(d: D) -> Result<Option<Option<String>>, D::Error> {
    Ok(Some(Option::<String>::deserialize(d)?))
}

pub async fn put(State(state): State<AppState>, Json(u): Json<Update>) -> ApiResult {
    blocking(move || {
        let bad = |m: String| ApiError::new(StatusCode::BAD_REQUEST, "invalid", m);
        if u.crossref_email.is_some() || u.crossref_email_prompted.is_some() {
            let mut g = config::load_global()?;
            if let Some(email) = u.crossref_email {
                g.crossref_email = match email.as_deref().map(str::trim) {
                    None | Some("") => None,
                    Some(e) => Some(config::validate_email(e).map_err(|e| bad(e.to_string()))?),
                };
                g.crossref_email_prompted = true;
                localrag_core::metadata::crossref::shared().set_email(g.crossref_email.as_deref());
            }
            if let Some(p) = u.crossref_email_prompted {
                g.crossref_email_prompted = p;
            }
            config::save_global(&g)?;
        }
        if u.chat_model.is_some() || u.num_ctx.is_some() || u.ocr.is_some() || u.think.is_some() || u.repair_quotes.is_some() || u.strict.is_some() {
            let mut ws = state.workspace()?;
            if let Some(m) = u.chat_model.filter(|m| !m.trim().is_empty()) {
                ws.config.chat_model = m;
            }
            if let Some(map) = u.num_ctx {
                for (model, n) in map {
                    if !(2048..=262_144).contains(&n) {
                        return Err(bad(format!("num_ctx for {model} must be between 2,048 and 262,144")));
                    }
                    ws.config.num_ctx.insert(model, n);
                }
            }
            if let Some(o) = u.ocr {
                ws.config.ocr = o;
            }
            if let Some(t) = u.think {
                ws.config.think = t;
            }
            if let Some(r) = u.repair_quotes {
                ws.config.repair_quotes = r;
            }
            if let Some(st) = u.strict {
                ws.config.strict = st;
            }
            ws.save_config()?;
        }
        Ok(Json(json!({ "saved": true })))
    })
    .await
}
