//! Crossref REST client (docs/ARCHITECTURE.md §7): DOI lookups and bibliographic queries,
//! one shared rate limiter per request type, polite pool when an email is set, a timestamped
//! response cache keyed by the request, and backoff on 429/5xx.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

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
    /// Plain-text abstract, when the publisher deposited one (Crossref sends JATS XML).
    #[serde(rename = "abstract", default)]
    pub abstract_text: Option<String>,
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
            "abstract": self.abstract_text,
        });
        if let Some(y) = self.year {
            v["issued"] = serde_json::json!({ "date-parts": [[y]] });
        }
        v
    }

    fn from_crossref(m: &Value) -> Option<Work> {
        // Titles may carry markup ("<scp>DNA</scp>", "<i>in vivo</i>"): plain text only.
        let first = |k: &str| m[k].as_array().and_then(|a| a.first()).and_then(Value::as_str).map(|t| jats_to_text(t).replace("\n\n", " "));
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
            abstract_text: m["abstract"].as_str().map(jats_to_text).filter(|a| !a.is_empty()),
        })
    }
}

/// JATS abstract markup to plain text: tags dropped, a leading "Abstract" heading removed,
/// paragraphs separated by blank lines, whitespace collapsed.
pub fn jats_to_text(s: &str) -> String {
    let mut out = String::new();
    let mut rest = s;
    while let Some(lt) = rest.find('<') {
        out.push_str(&rest[..lt]);
        let Some(gt) = rest[lt..].find('>') else {
            rest = "";
            break;
        };
        let tag = &rest[lt + 1..lt + gt];
        let name = tag.trim_start_matches('/').split([' ', '/']).next().unwrap_or("");
        if name.ends_with(":p") || name == "p" || name.ends_with(":sec") || name.ends_with(":title") {
            out.push_str("\n\n");
        } else if !tag.starts_with('/') {
            out.push(' ');
        }
        rest = &rest[lt + gt + 1..];
    }
    out.push_str(rest);
    let text = unescape(&out);
    let paras: Vec<String> = text.split("\n\n").map(|p| p.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|p| !p.is_empty()).collect();
    let paras: Vec<String> = paras.into_iter().enumerate().filter(|(i, p)| !(*i == 0 && p.eq_ignore_ascii_case("abstract"))).map(|(_, p)| p).collect();
    paras.join("\n\n")
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
    email: Mutex<Option<String>>,
    /// Cached responses younger than this are used instead of asking Crossref again.
    refresh_interval: Mutex<Duration>,
    /// One lock per request in flight, so identical concurrent requests wait for the first
    /// and then read its cached response instead of all going to Crossref.
    in_flight: Mutex<HashMap<String, Arc<Mutex<()>>>>,
}

/// How long cached responses are used before Crossref is asked again (`cache_refresh_days`).
pub const DEFAULT_REFRESH_DAYS: u32 = 14;

/// One cache file: the response and when it was fetched, keyed by the request.
#[derive(Serialize, Deserialize)]
struct CacheEntry {
    /// The request ("GET /works?query.bibliographic=…&rows=3"), without the email.
    request: String,
    /// Seconds since the Unix epoch.
    fetched_at: u64,
    /// The response body; `null` for a 404.
    response: Value,
}

/// One client per process, so the limits hold machine-wide across workspaces and jobs.
pub fn shared() -> &'static Crossref {
    static CLIENT: OnceLock<Crossref> = OnceLock::new();
    CLIENT.get_or_init(|| {
        let base = std::env::var("LOCALRAG_CROSSREF_URL").unwrap_or_else(|_| "https://api.crossref.org".into());
        Crossref::new(&base, cache_dir(), Arc::new(RealClock::default()))
    })
}

/// The shared client set up from the machine-wide config: polite-pool email and how long
/// cached responses are reused.
pub fn configured(g: &crate::config::GlobalConfig) -> &'static Crossref {
    let c = shared();
    c.set_email(g.crossref_email.as_deref());
    c.set_refresh_interval(Duration::from_secs(u64::from(g.cache_refresh_interval_days) * 86_400));
    c
}

/// `crossref/` under the per-user cache folder (`config::cache_dir`).
pub fn cache_dir() -> PathBuf {
    crate::config::cache_dir().join("crossref")
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
            email: Mutex::new(None),
            refresh_interval: Mutex::new(Duration::from_secs(u64::from(DEFAULT_REFRESH_DAYS) * 86_400)),
            in_flight: Mutex::new(HashMap::new()),
        }
    }

    /// How old a cached response may be before it's fetched again. Zero always fetches.
    pub fn set_refresh_interval(&self, interval: Duration) {
        *self.refresh_interval.lock().expect("refresh interval") = interval;
    }

    pub fn refresh_interval(&self) -> Duration {
        *self.refresh_interval.lock().expect("refresh interval")
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
        let body = self.get(&self.single, &format!("/works/{}", encode_path(&doi)), &[], Some(&format!("doi:{doi}")), refresh)?;
        Ok(body.and_then(|v| Work::from_crossref(&v["message"])))
    }

    /// Best bibliographic matches for free text (title, authors, year).
    pub fn search(&self, query: &str, rows: usize, refresh: bool) -> anyhow::Result<Vec<Work>> {
        let query = query.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        let rows_s = rows.to_string();
        let legacy = format!("query:{rows}:{query}");
        let body = self.get(&self.list, "/works", &[("query.bibliographic", query.as_str()), ("rows", rows_s.as_str())], Some(&legacy), refresh)?;
        Ok(body
            .and_then(|v| v["message"]["items"].as_array().cloned())
            .unwrap_or_default()
            .iter()
            .filter_map(Work::from_crossref)
            .collect())
    }

    /// GET through the cache: a response cached for this request less than the refresh
    /// interval ago is returned as is; otherwise (or with `refresh`) Crossref is asked and the
    /// cache entry replaced. `Ok(None)` is a 404. `legacy` is the key of the cache format
    /// before entries carried their request and timestamp; those still count, dated by mtime.
    fn get(&self, limiter: &Limiter, path: &str, query: &[(&str, &str)], legacy: Option<&str>, refresh: bool) -> anyhow::Result<Option<Value>> {
        let request = request_key(path, query);
        let cache_file = self.cache_dir.join(format!("{}.json", &sha256_str(&request)[..32]));
        if !refresh {
            if let Some(v) = self.fresh(&cache_file, &request) {
                return Ok(v);
            }
            if let Some((v, fetched_at)) = legacy.and_then(|k| self.fresh_legacy(k)) {
                cache_at(&cache_file, &request, v.clone().unwrap_or(Value::Null), fetched_at);
                return Ok(v);
            }
        }
        // Identical requests in flight wait here, then find the response the first one cached.
        let lock = self.in_flight.lock().expect("in flight").entry(request.clone()).or_default().clone();
        let result = {
            let _guard = lock.lock().expect("request lock");
            match self.fresh(&cache_file, &request) {
                Some(v) if !refresh => Ok(v),
                // Crossref unreachable or failing: an expired entry beats nothing.
                _ => self.fetch(limiter, path, query, &request, &cache_file).or_else(|e| self.cached(&cache_file, &request, None).ok_or(e)),
            }
        };
        let mut map = self.in_flight.lock().expect("in flight");
        if Arc::strong_count(&lock) <= 2 {
            map.remove(&request);
        }
        result
    }

    /// The cached response for `request` if it's younger than the refresh interval.
    /// `Some(None)` is a cached 404.
    fn fresh(&self, file: &std::path::Path, request: &str) -> Option<Option<Value>> {
        self.cached(file, request, Some(self.refresh_interval()))
    }

    /// The cached response for `request`, if any and (with `max_age`) not older than that.
    fn cached(&self, file: &std::path::Path, request: &str, max_age: Option<Duration>) -> Option<Option<Value>> {
        let text = std::fs::read_to_string(file).ok()?;
        let entry: CacheEntry = serde_json::from_str(&text).ok()?;
        if entry.request != request {
            return None; // hash collision, or an entry from an older format
        }
        let age = Duration::from_secs(now_secs().saturating_sub(entry.fetched_at));
        if max_age.is_some_and(|m| age >= m) {
            return None;
        }
        Some(if entry.response.is_null() { None } else { Some(entry.response) })
    }

    /// A pre-timestamp cache file (the bare response, named by the old key), dated by its
    /// mtime. Returned with that time so it can be rewritten in the current format.
    fn fresh_legacy(&self, key: &str) -> Option<(Option<Value>, u64)> {
        let file = self.cache_dir.join(format!("{}.json", &sha256_str(key)[..32]));
        let fetched_at = std::fs::metadata(&file).ok()?.modified().ok()?.duration_since(UNIX_EPOCH).ok()?.as_secs();
        let v: Value = serde_json::from_str(&std::fs::read_to_string(&file).ok()?).ok()?;
        if v.get("request").is_some() && v.get("fetched_at").is_some() {
            return None; // already a current-format entry that happens to share the name
        }
        let _ = std::fs::remove_file(&file);
        if Duration::from_secs(now_secs().saturating_sub(fetched_at)) >= self.refresh_interval() {
            return None;
        }
        Some((if v.is_null() { None } else { Some(v) }, fetched_at))
    }

    /// Asks Crossref (limiter, polite-pool identification, backoff) and caches the response.
    fn fetch(&self, limiter: &Limiter, path: &str, query: &[(&str, &str)], request: &str, cache_file: &std::path::Path) -> anyhow::Result<Option<Value>> {
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
                cache(cache_file, request, Value::Null);
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
            cache(cache_file, request, v.clone());
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

/// The request a cache entry belongs to: path and parameters in a fixed order, without the
/// email (it identifies us, it doesn't change the answer).
fn request_key(path: &str, query: &[(&str, &str)]) -> String {
    let mut params: Vec<String> = query.iter().map(|(k, v)| format!("{k}={v}")).collect();
    params.sort();
    if params.is_empty() { format!("GET {path}") } else { format!("GET {path}?{}", params.join("&")) }
}

fn now_secs() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

fn cache(file: &std::path::Path, request: &str, response: Value) {
    cache_at(file, request, response, now_secs());
}

/// Writes the entry atomically (temp file, then rename), so readers never see half of one.
fn cache_at(file: &std::path::Path, request: &str, response: Value, fetched_at: u64) {
    if let Some(dir) = file.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let entry = CacheEntry { request: request.to_string(), fetched_at, response };
    let tmp = file.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&tmp, serde_json::to_vec(&entry).unwrap_or_default()).is_ok() {
        let _ = std::fs::rename(&tmp, file);
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jats_abstracts_become_plain_text() {
        assert_eq!(jats_to_text("<jats:title>Abstract</jats:title><jats:sec><jats:title>Background</jats:title><jats:p>Rates  rose.</jats:p></jats:sec>"), "Background\n\nRates rose.");
        assert_eq!(jats_to_text("Plain &amp; simple"), "Plain & simple");
        assert_eq!(request_key("/works", &[("rows", "3"), ("query.bibliographic", "x y")]), "GET /works?query.bibliographic=x y&rows=3");
    }
}
