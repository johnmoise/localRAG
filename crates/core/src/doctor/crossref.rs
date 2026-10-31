use anyhow::{Context, ensure};
use serde_json::Value;

use super::Outcome;

const API: &str = "https://api.crossref.org";
const TEST_DOI: &str = "10.1038/nature14539"; // LeCun, Bengio & Hinton (2015), "Deep learning"

/// One single-record and one list request, in the polite pool when an email is set (§7).
/// Two sequential requests stay well inside even the public pool's limits.
pub(super) async fn check(client: &reqwest::Client, email: Option<&str>) -> anyhow::Result<Outcome> {
    let ua = match email {
        Some(e) => format!("localrag/{} (mailto:{e})", env!("CARGO_PKG_VERSION")),
        None => format!("localrag/{}", env!("CARGO_PKG_VERSION")),
    };
    let mailto: Vec<(&str, &str)> = email.map(|e| vec![("mailto", e)]).unwrap_or_default();

    let resp = client
        .get(format!("{API}/works/{TEST_DOI}"))
        .query(&mailto)
        .header(reqwest::header::USER_AGENT, &ua)
        .send()
        .await?
        .error_for_status()?;
    let limit_header = header(&resp, "x-rate-limit-limit");
    let interval_header = header(&resp, "x-rate-limit-interval");
    let single: Value = resp.json().await?;
    let title = single["message"]["title"][0].as_str().context("no title in DOI record")?;
    ensure!(title.eq_ignore_ascii_case("Deep learning"), "unexpected title {title:?}");

    let list: Value = client
        .get(format!("{API}/works"))
        .query(&[("query.bibliographic", "LeCun Bengio Hinton deep learning 2015"), ("rows", "1")])
        .query(&mailto)
        .header(reqwest::header::USER_AGENT, &ua)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    let top_doi = list["message"]["items"][0]["DOI"].as_str().unwrap_or("");

    let pool = if email.is_some() { "polite" } else { "public (no email)" };
    let detail = format!(
        "{pool} pool; DOI lookup OK; query top hit {top_doi}; rate-limit headers: {limit_header} per {interval_header}"
    );
    if top_doi.eq_ignore_ascii_case(TEST_DOI) {
        Ok(Outcome::Pass(detail))
    } else {
        Ok(Outcome::Warn(format!("{detail} (expected {TEST_DOI} as the top hit)")))
    }
}

fn header(resp: &reqwest::Response, name: &str) -> String {
    resp.headers().get(name).and_then(|v| v.to_str().ok()).unwrap_or("none").to_string()
}
