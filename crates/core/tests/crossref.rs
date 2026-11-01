//! The Crossref client against a local mock server (ARCHITECTURE.md §7).

use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use localrag_core::metadata::crossref::{Crossref, POLITE_LIST, POLITE_SINGLE, PUBLIC_SINGLE, Pool};
use localrag_core::metadata::ratelimit::RealClock;

#[derive(Debug, Clone)]
struct Seen {
    at: Instant,
    target: String,
    user_agent: String,
}

/// (status, extra headers, body) for a request target and how many times it was requested.
type Reply = (u16, Vec<(&'static str, String)>, String);

/// Serves canned Crossref responses and records every request.
fn mock(behaviour: fn(&str, usize) -> Reply) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let seen: Arc<Mutex<Vec<Seen>>> = Arc::default();
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let mut stream = stream.unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut request_line = String::new();
            reader.read_line(&mut request_line).unwrap();
            let target = request_line.split_whitespace().nth(1).unwrap_or("").to_string();
            let mut user_agent = String::new();
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap() == 0 || line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_lowercase().strip_prefix("user-agent:") {
                    user_agent = v.trim().to_string();
                }
            }
            let n = {
                let mut l = log.lock().unwrap();
                l.push(Seen { at: Instant::now(), target: target.clone(), user_agent });
                l.iter().filter(|s| s.target == target).count()
            };
            let (status, headers, body) = behaviour(&target, n);
            let mut resp = format!("HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
            for (k, v) in headers {
                resp.push_str(&format!("{k}: {v}\r\n"));
            }
            resp.push_str("\r\n");
            resp.push_str(&body);
            let _ = stream.write_all(resp.as_bytes());
        }
    });
    (base, seen)
}

fn work_json(doi: &str, title: &str) -> String {
    serde_json::json!({ "message": {
        "DOI": doi, "title": [title], "author": [{ "family": "Parry", "given": "Carla" }],
        "issued": { "date-parts": [[2011, 10]] }, "container-title": ["Biomarkers &amp; Prevention"], "page": "1996-2005", "type": "journal-article", "is-referenced-by-count": 42
    }})
    .to_string()
}

fn client(base: &str) -> (Crossref, tempfile::TempDir) {
    let cache = tempfile::tempdir().unwrap();
    (Crossref::new(base, cache.path().to_path_buf(), Arc::new(RealClock::default())), cache)
}

#[test]
fn doi_lookup_parses_work_and_identifies_politely() {
    let (base, seen) = mock(|target, _| {
        if target.starts_with("/works/10.1158") {
            (200, vec![], work_json("10.1158/1055-9965.EPI-11-0472", "Cancer Survivors: A Booming Population"))
        } else {
            (404, vec![], "{}".into())
        }
    });
    let (c, _cache) = client(&base);
    assert_eq!(c.pool(), Pool::Public);
    c.set_email(Some("me@example.org"));
    assert_eq!(c.pool(), Pool::Polite);
    assert_eq!(c.limits(), (POLITE_SINGLE, POLITE_LIST));

    let w = c.work("10.1158/1055-9965.EPI-11-0472", false).unwrap().unwrap();
    assert_eq!(w.doi, "10.1158/1055-9965.epi-11-0472");
    assert_eq!(w.year, Some(2011));
    assert_eq!(w.container.as_deref(), Some("Biomarkers & Prevention"), "HTML entities decoded");
    assert_eq!(w.page.as_deref(), Some("1996-2005"));

    let s = seen.lock().unwrap().clone();
    assert!(s[0].target.contains("mailto=me%40example.org"), "{}", s[0].target);
    assert!(s[0].target.starts_with("/works/10.1158%2F1055"), "DOI escaped in the path: {}", s[0].target);
    assert!(s[0].user_agent.contains("mailto:me@example.org"), "{}", s[0].user_agent);

    assert!(c.work("10.9999/unknown", false).unwrap().is_none(), "404 → None");
    // Cached: no second request for the same DOI.
    c.work("10.1158/1055-9965.EPI-11-0472", false).unwrap();
    assert_eq!(seen.lock().unwrap().len(), 2);
    // Refresh bypasses the cache.
    c.work("10.1158/1055-9965.EPI-11-0472", true).unwrap();
    assert_eq!(seen.lock().unwrap().len(), 3);

    c.set_email(None);
    assert_eq!(c.limits().0, PUBLIC_SINGLE, "clearing the email returns to public limits");
}

#[test]
fn public_pool_spacing_and_server_announced_limits_are_respected() {
    // Announces 2 requests per second: stricter than the public pool's 5.
    let (base, seen) = mock(|_, _| (200, vec![("X-Rate-Limit-Limit", "2".into()), ("X-Rate-Limit-Interval", "1s".into())], work_json("10.1/x", "T")));
    let (c, _cache) = client(&base);
    let started = Instant::now();
    for i in 0..5 {
        c.work(&format!("10.1234/{i}"), false).unwrap();
    }
    let s = seen.lock().unwrap();
    let gaps: Vec<Duration> = s.windows(2).map(|w| w[1].at - w[0].at).collect();
    // First gap at the pool's 5/s (0.2 s); after the header, 2/s (0.5 s).
    assert!(gaps[0] >= Duration::from_millis(190), "{gaps:?}");
    assert!(gaps[1..].iter().all(|g| *g >= Duration::from_millis(490)), "{gaps:?}");
    assert!(started.elapsed() >= Duration::from_millis(1600));
}

#[test]
fn rate_limited_responses_are_retried_after_the_server_says() {
    let (base, seen) = mock(|_, n| {
        if n == 1 { (429, vec![("Retry-After", "1".into())], "{}".into()) } else { (200, vec![], work_json("10.1/x", "T")) }
    });
    let (c, _cache) = client(&base);
    let started = Instant::now();
    assert!(c.work("10.5555/retry", false).unwrap().is_some());
    assert_eq!(seen.lock().unwrap().len(), 2);
    assert!(started.elapsed() >= Duration::from_secs(1), "waited Retry-After");
}

#[test]
fn references_are_enriched_by_doi_or_citation_search_else_kept_as_stubs() {
    let (base, _seen) = mock(|target, _| {
        if target.starts_with("/works/10.1000%2Fknown") {
            (200, vec![], work_json("10.1000/known", "A Known Work"))
        } else if target.contains("query.bibliographic=Parry") {
            (200, vec![], serde_json::json!({ "message": { "items": [serde_json::from_str::<serde_json::Value>(&work_json("10.1158/x", "Cancer Survivors: A Booming Population")).unwrap()["message"]] } }).to_string())
        } else {
            (200, vec![], serde_json::json!({ "message": { "items": [] } }).to_string())
        }
    });
    let (c, _cache) = client(&base);
    let dir = tempfile::tempdir().unwrap();
    let conn = localrag_core::db::open(&dir.path().join("index.db")).unwrap();
    conn.execute_batch(
        "INSERT INTO documents (id, path, size, mtime_ns, sha256, status, kind, parser_version, chunker_version) VALUES (1, 'a.pdf', 1, 1, 'x', 'indexed', 'pdf', 1, 2);
         INSERT INTO doc_references (doc_id, ord, raw, doi) VALUES
           (1, 0, 'Someone A. A known work. J. 2011. doi:10.1000/known', '10.1000/known'),
           (1, 1, 'Parry C, Kent EE. Cancer survivors: a booming population. Cancer Epidemiol Biomarkers Prev. 2011;20:1996–2005.', NULL),
           (1, 2, 'Nobody Z. An obscure report nobody indexed. Local Press. 1987.', NULL);",
    ).unwrap();
    let r = localrag_core::references::enrich_pending(&conn, &c, &mut |_, _| {}).unwrap();
    assert_eq!((r.matched, r.stubs), (2, 1));
    let rows: Vec<(String, Option<String>, Option<i64>)> = conn
        .prepare("SELECT status, doi, cited_by FROM doc_references ORDER BY ord").unwrap()
        .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?))).unwrap().collect::<Result<_, _>>().unwrap();
    assert_eq!(rows[0].0, "matched");
    assert_eq!((rows[1].0.as_str(), rows[1].1.as_deref()), ("matched", Some("10.1158/x")));
    assert_eq!(rows[1].2, Some(42), "Crossref's is-referenced-by-count");
    assert_eq!(rows[2].0, "stub");
}
