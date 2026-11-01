//! Integrity of a brain (§13): every `[FACT]` line against the registry, and every fact's
//! citations against the index. Runs on every read, organize and write, and on `check`.

use std::collections::{HashMap, HashSet};

use regex::Regex;
use rusqlite::{Connection, params};
use serde::Serialize;

use super::file::{Brain, FactCitation, FactLine, Line};
use super::registry;
use crate::db;
use crate::refs::{self, Resolution};
use crate::scan::sha256_str;

#[derive(Debug, Clone, Serialize)]
pub struct CitationView {
    pub label: String,
    pub quote: String,
    pub doc_sha256: String,
    pub chunk_text_sha: String,
    pub resolution: Resolution,
    /// The chunk the quote is in now (for opening the source).
    pub chunk_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FactView {
    pub id: String,
    pub claim: String,
    pub citations: Vec<CitationView>,
    /// active (usable), stale (a source was removed), broken (a quote is gone),
    /// unresolved (the line itself failed a registry check).
    pub status: String,
    pub problem: Option<String>,
    pub conflicts: Vec<String>,
}

impl FactView {
    pub fn usable(&self) -> bool {
        self.status == "active"
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LineView {
    Heading { level: usize, text: String },
    Thought { text: String, unknown_ids: Vec<String> },
    Fact(FactView),
    Blank,
    Invalid { raw: String, error: String },
}

#[derive(Debug, Clone, Serialize)]
pub struct Problem {
    /// invalid_line, unknown_id, duplicate_id, deleted_id, edited, missing, dangling_mention, stale, broken
    pub kind: String,
    pub fact_id: Option<String>,
    /// 0-based line number in the file.
    pub line: Option<usize>,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct Checked {
    pub model: String,
    pub hash: String,
    pub lines: Vec<LineView>,
    pub problems: Vec<Problem>,
}

impl Checked {
    /// Usable facts, in file order.
    pub fn usable(&self) -> Vec<&FactView> {
        self.lines.iter().filter_map(|l| if let LineView::Fact(f) = l { Some(f) } else { None }).filter(|f| f.usable()).collect()
    }

    pub fn ok(&self) -> bool {
        self.problems.is_empty()
    }
}

fn mention_re() -> &'static Regex {
    static RE: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();
    RE.get_or_init(|| Regex::new(r"\bF-\d{4,}\b").unwrap())
}

/// Resolves one fact's citations, recording each result. The fact's status follows from the
/// worst one: broken > stale (source removed) > active.
pub fn resolve_citations(conn: &Connection, model: &str, id: Option<&str>, citations: &[FactCitation]) -> anyhow::Result<(Vec<CitationView>, &'static str)> {
    let mut views = Vec::new();
    let mut status = "active";
    for (i, c) in citations.iter().enumerate() {
        let r = refs::resolve(conn, &c.doc_sha256, &c.chunk_text_sha, Some(&c.quote))?;
        match r.resolution {
            Resolution::Broken => status = "broken",
            Resolution::SourceRemoved if status != "broken" => status = "stale",
            _ => {}
        }
        if let Some(id) = id {
            conn.execute(
                "UPDATE fact_citations SET resolution = ?4, resolved_chunk_id = ?5, checked_at = ?6 WHERE model = ?1 AND fact_id = ?2 AND ord = ?3",
                params![model, id, i as i64, serde_json::to_value(r.resolution)?.as_str(), r.chunk_id, db::now_rfc3339()],
            )?;
        }
        views.push(CitationView {
            label: c.label.clone(),
            quote: c.quote.clone(),
            doc_sha256: c.doc_sha256.clone(),
            chunk_text_sha: c.chunk_text_sha.clone(),
            resolution: r.resolution,
            chunk_id: r.chunk_id,
        });
    }
    Ok((views, status))
}

/// Checks a brain against the registry and the index, updating fact statuses.
pub fn check(conn: &Connection, brain: &Brain) -> anyhow::Result<Checked> {
    let model = &brain.model;
    let registry: HashMap<String, registry::Registered> = registry::all(conn, model)?.into_iter().map(|f| (f.id.clone(), f)).collect();
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for l in &brain.lines {
        if let Line::Fact { fact, .. } = l {
            *counts.entry(fact.id.as_str()).or_default() += 1;
        }
    }
    let mut problems = Vec::new();
    let mut lines = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for (n, l) in brain.lines.iter().enumerate() {
        let view = match l {
            Line::Heading { level, text } => LineView::Heading { level: *level, text: text.clone() },
            Line::Blank => LineView::Blank,
            Line::Invalid { raw, error } => {
                problems.push(Problem { kind: "invalid_line".into(), fact_id: None, line: Some(n), message: format!("Line {}: {error}; ignored.", n + 1) });
                LineView::Invalid { raw: raw.clone(), error: error.clone() }
            }
            Line::Thought { text } => {
                let unknown: Vec<String> = mention_re().find_iter(text).map(|m| m.as_str().to_string()).filter(|id| registry.get(id).is_none_or(|f| f.status == "deleted")).collect();
                for id in &unknown {
                    problems.push(Problem { kind: "dangling_mention".into(), fact_id: Some(id.clone()), line: Some(n), message: format!("A thought on line {} mentions {id}, which isn't an active fact.", n + 1) });
                }
                LineView::Thought { text: text.clone(), unknown_ids: unknown }
            }
            Line::Fact { fact, raw } => {
                seen.insert(fact.id.clone());
                LineView::Fact(check_fact(conn, model, n, fact, raw, registry.get(&fact.id), counts[fact.id.as_str()], &mut problems)?)
            }
        };
        lines.push(view);
    }
    for f in registry.values() {
        if f.status != "deleted" && !seen.contains(&f.id) {
            problems.push(Problem { kind: "missing".into(), fact_id: Some(f.id.clone()), line: None, message: format!("{} is approved but missing from the file: restore it or delete it.", f.id) });
        }
    }
    Ok(Checked { model: model.clone(), hash: brain.hash.clone(), lines, problems })
}

#[allow(clippy::too_many_arguments)]
fn check_fact(conn: &Connection, model: &str, n: usize, fact: &FactLine, raw: &str, reg: Option<&registry::Registered>, copies: usize, problems: &mut Vec<Problem>) -> anyhow::Result<FactView> {
    let mut view = FactView { id: fact.id.clone(), claim: fact.claim.clone(), citations: Vec::new(), status: "unresolved".into(), problem: None, conflicts: Vec::new() };
    let mut fail = |kind: &str, msg: String, view: &mut FactView| {
        view.problem = Some(msg.clone());
        problems.push(Problem { kind: kind.into(), fact_id: Some(fact.id.clone()), line: Some(n), message: msg });
    };
    let Some(reg) = reg else {
        fail("unknown_id", format!("{} on line {} isn't a registered fact; excluded.", fact.id, n + 1), &mut view);
        view.citations = resolve_citations(conn, model, None, &fact.citations)?.0;
        return Ok(view);
    };
    if reg.status == "deleted" {
        fail("deleted_id", format!("{} was deleted and can't come back; excluded.", fact.id), &mut view);
        return Ok(view);
    }
    if copies > 1 {
        fail("duplicate_id", format!("{} appears {copies} times; all copies are excluded until only one is left.", fact.id), &mut view);
        return Ok(view);
    }
    let edited = sha256_str(raw) != reg.line_sha;
    let (citations, status) = resolve_citations(conn, model, if edited { None } else { Some(&fact.id) }, &fact.citations)?;
    view.citations = citations;
    if edited {
        // Edited outside localrag: accepted as a hand edit only if every quote still verifies.
        if status == "active" {
            registry::rewrite(conn, model, fact)?;
            resolve_citations(conn, model, Some(&fact.id), &fact.citations)?;
        } else {
            fail("edited", format!("{} was edited outside localrag and its quotes don't verify; excluded until fixed.", fact.id), &mut view);
            return Ok(view);
        }
    }
    registry::set_status(conn, model, &fact.id, status)?;
    view.status = status.into();
    match status {
        "stale" => fail("stale", format!("{}: a cited document was removed from corpus/; excluded until it's back.", fact.id), &mut view),
        "broken" => fail("broken", format!("{}: a quote is no longer in the corpus; find it again or delete the fact.", fact.id), &mut view),
        _ => {}
    }
    view.conflicts = registry::conflicts(conn, model, &fact.id)?;
    Ok(view)
}
