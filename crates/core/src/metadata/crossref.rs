//! Crossref REST client (ARCHITECTURE.md §7): DOI lookups and bibliographic queries, one
//! shared rate limiter per request type, polite pool when an email is set, a response cache,
//! and backoff on 429/5xx.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::ratelimit::{Clock, Limiter, Limits, RealClock};
use crate::scan::sha256_str;

/// Crossref's published limits.
pub const POLITE_SINGLE: Limits = Limits { per_second: 10.0, concurrency: 3 };
pub const POLITE_LIST: Limits = Limits { per_second: 3.0, concurrency: 3 };
pub const PUBLIC_SINGLE: Limits = Limits { per_second: 5.0, concurrency: 1 };
pub const PUBLIC_LIST: Limits = Limits { per_second: 1.0, concurrency: 1 };

/// Bibliographic data for one work, in CSL-JSON field names.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Work {
    #[serde(rename = "DOI")]
    pub doi: String,
    pub title: Option<String>,
    pub author: Vec<Author>,
    /// Publication year (`issued`).
    pub year: Option<i32>,
    #[serde(rename = "container-title")]
    pub container: Option<String>,
    /// e.g. "1996-2005"
    pub page: Option<String>,
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// How many works cite this one (Crossref's `is-referenced-by-count`).
    #[serde(rename = "is-referenced-by-count", default)]
    pub cited_by: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Author {
    pub family: Option<String>,
    pub given: Option<String>,
    /// Organisations ("name" in Crossref).
    pub literal: Option<String>,
}

impl Work {
    /// CSL-JSON as stored in `documents.csl_json`.
    pub fn to_csl(&self) -> Value {
        let mut v = serde_json::json!({
            "DOI": self.doi,
            "title": self.title,
            "author": self.author,
            "container-title": self.container,
            "page": self.page,
            "type": self.kind,
            "is-referenced-by-count": self.cited_by,
        });
        if let Some(y) = self.year {
            v["issued"] = serde_json::json!({ "date-parts": [[y]] });
        }
        v
    }

    fn from_crossref(m: &Value) -> Option<Work> {
        let first = |k: &str| m[k].as_array().and_then(|a| a.first()).and_then(Value::as_str).map(unescape);
        Some(Work {
            doi: m["DOI"].as_str()?.to_lowercase(),
            title: first("title"),
            author: m["author"]
                .as_array()
                .map(|a| {
                    a.iter()
                        .map(|p| Author {
                            family: p["family"].as_str().map(unescape),
                            given: p["given"].as_str().map(unescape),
                            literal: p["name"].as_str().map(unescape),
                        })
                        .collect()
                })
                .unwrap_or_default(),
            year: ["issued", "published-print", "published-online", "created"]
                .iter()
                .find_map(|k| m[*k]["date-parts"][0][0].as_i64())
                .map(|y| y as i32),
            container: first("container-title"),
            page: m["page"].as_str().map(str::to_string),
            kind: m["type"].as_str().map(str::to_string),
            cited_by: m["is-referenced-by-count"].as_i64(),
        })
    }
}

/// Crossref returns some fields with HTML entities ("Biomarkers &amp; Prevention").
fn unescape(s: &str) -> String {
    s.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#39;", "'")
}

pub struct Crossref {
    base: String,
    http: reqwest::blocking::Client,
    single: Limiter,
    list: Limiter,
    cache_dir: PathBuf,
    email: std::sync::Mutex<Option<String>>,
}

/// One client per process, so the limits hold machine-wide across workspaces and jobs.
pub fn shared() -> &'static Crossref {
    static CLIENT: OnceLock<Crossref> = OnceLock::new();
    CLIENT.get_or_init(|| {
        let base = std::env::var("LOCALRAG_CROSSREF_URL").unwrap_or_else(|_| "https://api.crossref.org".into());
        Crossref::new(&base, cache_dir(), Arc::new(RealClock::default()))
    })
}

/// `$LOCALRAG_CACHE_DIR/crossref`, else `~/.cache/localrag/crossref`.
pub fn cache_dir() -> PathBuf {
    let base = std::env::var_os("LOCALRAG_CACHE_DIR").map(PathBuf::from).unwrap_or_else(|| {
        std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(".cache").join("localrag")
    });
    base.join("crossref")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pool {
    Polite,
    Public,
}

impl Crossref {
    pub fn new(base: &str, cache_dir: PathBuf, clock: Arc<dyn Clock>) -> Self {
        let http = reqwest::blocking::Client::builder().timeout(Duration::from_secs(30)).build().expect("http client");
        Crossref {
            base: base.trim_end_matches('/').to_string(),
            http,
            single: Limiter::new(PUBLIC_SINGLE, clock.clone()),
            list: Limiter::new(PUBLIC_LIST, clock),
            cache_dir,
            email: std::sync::Mutex::new(None),
        }
    }

    /// Sets (or clears) the polite-pool email, switching both limiters to that pool's limits.
    pub fn set_email(&self, email: Option<&str>) {
        let email = email.map(str::trim).filter(|e| !e.is_empty()).map(str::to_string);
        let mut current = self.email.lock().expect("email");
        if *current == email {
            return;
        }
        let polite = email.is_some();
        *current = email;
        self.single.set_limits(if polite { POLITE_SINGLE } else { PUBLIC_SINGLE });
        self.list.set_limits(if polite { POLITE_LIST } else { PUBLIC_LIST });
    }

    pub fn pool(&self) -> Pool {
        if self.email.lock().expect("email").is_some() { Pool::Polite } else { Pool::Public }
    }

    pub fn limits(&self) -> (Limits, Limits) {
        (self.single.limits(), self.list.limits())
    }

    /// The work with this DOI, or `None` if Crossref doesn't know it.
    pub fn work(&self, doi: &str, refresh: bool) -> anyhow::Result<Option<Work>> {
        let doi = doi.trim().to_lowercase();
        let key = format!("doi:{doi}");
        let body = self.get(&self.single, &format!("/works/{}", encode_path(&doi)), &[], &key, refresh)?;
        Ok(body.and_then(|v| Work::from_crossref(&v["message"])))
    }

    /// Best bibliographic matches for free text (title, authors, year).
    pub fn search(&self, query: &str, rows: usize, refresh: bool) -> anyhow::Result<Vec<Work>> {
        let query = query.split_whitespace().collect::<Vec<_>>().join(" ");
        let key = format!("query:{rows}:{}", query.to_lowercase());
        let rows_s = rows.to_string();
        let body = self.get(&self.list, "/works", &[("query.bibliographic", query.as_str()), ("rows", rows_s.as_str())], &key, refresh)?;
        Ok(body
            .and_then(|v| v["message"]["items"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(Work::from_crossref)
            .collect())
    }

    /// GET with cache, limiter, polite-pool identification and backoff. `Ok(None)` is a 404.
    fn get(&self, limiter: &Limiter, path: &str, query: &[(&str, &str)], cache_key: &str, refresh: bool) -> anyhow::Result<Option<Value>> {
        let cache_file = self.cache_dir.join(format!("{}.json", &sha256_str(cache_key)[..32]));
        if !refresh && let Ok(text) = std::fs::read_to_string(&cache_file) {
            return Ok(serde_json::from_str(&text).ok());
        }
        let email = self.email.lock().expect("email").clone();
        let ua = match &email {
            Some(e) => format!("localrag/{} (mailto:{e})", env!("CARGO_PKG_VERSION")),
            None => format!("localrag/{}", env!("CARGO_PKG_VERSION")),
        };
        let mut params: Vec<(&str, &str)> = query.to_vec();
        if let Some(e) = &email {
            params.push(("mailto", e));
        }

        let mut delay = Duration::from_secs(1);
        for attempt in 0..4 {
            let resp = {
                let _permit = limiter.acquire();
                self.http.get(format!("{}{path}", self.base)).query(&params).header(reqwest::header::USER_AGENT, &ua).send()
            };
            let resp = match resp {
                Ok(r) => r,
                Err(e) if attempt < 3 => {
                    std::thread::sleep(delay);
                    delay *= 2;
                    let _ = e;
                    continue;
                }
                Err(e) => return Err(e).context("Crossref unreachable"),
            };
            adapt(limiter, resp.headers());
            let status = resp.status();
            if status.as_u16() == 404 {
                cache(&cache_file, &Value::Null);
                return Ok(None);
            }
            if status.as_u16() == 429 || status.is_server_error() {
                if attempt == 3 {
                    bail!("Crossref returned {status} after retries; will retry on a later sync");
                }
                let retry_after = resp
                    .headers()
                    .get(reqwest::header::RETRY_AFTER)
                    .and_then(|v| v.to_str().ok())
                    .and_then(|v| v.parse::<u64>().ok())
                    .map(Duration::from_secs);
                std::thread::sleep(retry_after.unwrap_or(delay) + jitter());
                delay *= 2;
                continue;
            }
            if !status.is_success() {
                bail!("Crossref returned {status}");
            }
            let v: Value = resp.json().context("Crossref response was not JSON")?;
            cache(&cache_file, &v);
            return Ok(Some(v));
        }
        unreachable!("loop returns")
    }
}

/// Follows stricter limits Crossref announces (`X-Rate-Limit-Limit`/`-Interval`, `X-Concurrency-Limit`).
fn adapt(limiter: &Limiter, h: &reqwest::header::HeaderMap) {
    let num = |k: &str| h.get(k).and_then(|v| v.to_str().ok()).map(str::trim).map(str::to_string);
    let limit = num("x-rate-limit-limit").and_then(|v| v.parse::<f64>().ok());
    let interval = num("x-rate-limit-interval").and_then(|v| v.trim_end_matches('s').parse::<f64>().ok()).filter(|s| *s > 0.0);
    let rate = match (limit, interval) {
        (Some(l), Some(i)) => Some(l / i),
        (Some(l), None) => Some(l),
        _ => None,
    };
    let conc = num("x-concurrency-limit").and_then(|v| v.parse::<usize>().ok());
    limiter.tighten(rate, conc);
}

fn cache(file: &std::path::Path, v: &Value) {
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(file, v.to_string());
}

fn jitter() -> Duration {
    let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
    Duration::from_millis((nanos % 250) as u64)
}

/// DOIs contain `/` and other characters that must be escaped inside a path segment.
fn encode_path(s: &str) -> String {
    s.bytes()
        .map(|b| if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") })
        .collect()
}
