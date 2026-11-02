//! Second brains (docs/ARCHITECTURE.md §12, §13): per reasoning model, a Markdown file of
//! approved, cited facts that the model reads before it researches a question.

pub mod file;
pub mod integrity;
pub mod organize;
pub mod queue;
pub mod registry;

use rusqlite::Connection;
use serde::Serialize;

use crate::embed::Embedder;
use crate::workspace::Workspace;
use file::{FactLine, Line};
use integrity::{Checked, FactView};

/// Rough token estimate for budgeting brain text (≈ 4 characters per token, rounded up).
fn tokens(s: &str) -> usize {
    s.chars().count().div_ceil(4)
}

/// "F-0012" → "F12" (the compact form the model sees).
pub fn short_id(id: &str) -> String {
    format!("F{}", file::id_number(id).unwrap_or(0))
}

/// "F12" or "F-0012" → "F-0012"
pub fn long_id(s: &str) -> Option<String> {
    let n: u32 = s.trim().trim_start_matches(['F', 'f']).trim_start_matches('-').parse().ok()?;
    Some(format!("F-{n:04}"))
}

fn compact(f: &FactView) -> String {
    let cites: Vec<&str> = f.citations.iter().map(|c| c.label.as_str()).collect();
    let conflict = if f.conflicts.is_empty() {
        String::new()
    } else {
        format!("⚠ conflicts with {}: ", f.conflicts.iter().map(|c| short_id(c)).collect::<Vec<_>>().join(", "))
    };
    format!("[{}] {conflict}{} ({})", short_id(&f.id), f.claim, cites.join("; "))
}

#[derive(Debug, Clone, Serialize)]
pub struct BrainRead {
    /// What the model sees.
    pub text: String,
    /// Fact IDs included.
    pub included: Vec<String>,
    pub warnings: Vec<String>,
}

pub const READ_HEADER: &str = "Known facts (verified; cite the original source shown, not the fact ID):";

/// `read_brain` (§12): only usable `[FACT]` lines, compact, in file order. If they don't fit
/// `budget_tokens`, the facts most similar to `query` are kept and a warning raised.
pub fn read(conn: &Connection, embedder: &dyn Embedder, checked: &Checked, query: &str, budget_tokens: usize) -> anyhow::Result<BrainRead> {
    let mut warnings: Vec<String> = Vec::new();
    let excluded = checked.lines.iter().filter(|l| matches!(l, integrity::LineView::Fact(f) if !f.usable())).count();
    if excluded > 0 {
        warnings.push(format!("{excluded} fact(s) excluded from the brain: they failed integrity checks (see Second Brain)."));
    }
    let usable = checked.usable();
    if usable.is_empty() {
        return Ok(BrainRead { text: "The brain has no facts yet.".into(), included: Vec::new(), warnings });
    }
    let lines: Vec<(String, String)> = usable.iter().map(|f| (f.id.clone(), compact(f))).collect();
    let total: usize = lines.iter().map(|(_, l)| tokens(l) + 1).sum();
    let keep: Vec<String> = if total <= budget_tokens {
        lines.iter().map(|(id, _)| id.clone()).collect()
    } else {
        let ranked = registry::nearest(conn, embedder, &checked.model, "fact", query, usable.len())?;
        let mut used = 0;
        let mut keep = Vec::new();
        for (id, _) in ranked {
            let Some((_, l)) = lines.iter().find(|(i, _)| *i == id) else { continue };
            if used + tokens(l) + 1 > budget_tokens {
                break;
            }
            used += tokens(l) + 1;
            keep.push(id);
        }
        warnings.push(format!("The brain doesn't fit its budget: {} of {} facts were included, chosen by relevance to the question.", keep.len(), lines.len()));
        keep
    };
    let mut text = String::from(READ_HEADER);
    let mut included = Vec::new();
    for (id, l) in &lines {
        if keep.contains(id) {
            text.push('\n');
            text.push_str(l);
            included.push(id.clone());
        }
    }
    Ok(BrainRead { text, included, warnings })
}

/// Loads and checks a brain in one go.
pub fn load_checked(ws: &Workspace, conn: &Connection, model: &str) -> anyhow::Result<(file::Brain, Checked)> {
    let b = file::load(ws, model)?;
    let c = integrity::check(conn, &b)?;
    Ok((b, c))
}

/// Inserts a fact line at the end of `## Unsorted` (created at the end if missing).
pub fn append_fact(lines: &mut Vec<Line>, fact: &FactLine) {
    let line = file::parse_line(&file::render_fact(fact));
    let start = lines.iter().position(|l| matches!(l, Line::Heading { level: 2, text } if text.eq_ignore_ascii_case("unsorted")));
    match start {
        Some(i) => {
            let end = lines[i + 1..].iter().position(|l| matches!(l, Line::Heading { .. })).map_or(lines.len(), |p| i + 1 + p);
            let mut at = end;
            while at > i + 1 && matches!(lines[at - 1], Line::Blank) {
                at -= 1;
            }
            lines.insert(at, line);
        }
        None => {
            if !matches!(lines.last(), Some(Line::Blank) | None) {
                lines.push(Line::Blank);
            }
            lines.push(Line::Heading { level: 2, text: "Unsorted".into() });
            lines.push(line);
        }
    }
}

/// Replaces the line of fact `id` (every copy) with `fact`'s line.
pub fn replace_fact(lines: &mut [Line], fact: &FactLine) -> bool {
    let new = file::parse_line(&file::render_fact(fact));
    let mut hit = false;
    for l in lines.iter_mut() {
        if matches!(l, Line::Fact { fact: f, .. } if f.id == fact.id) {
            *l = new.clone();
            hit = true;
        }
    }
    hit
}

/// Removes every line of fact `id`.
pub fn remove_fact(lines: &mut Vec<Line>, id: &str) -> bool {
    let before = lines.len();
    lines.retain(|l| !matches!(l, Line::Fact { fact, .. } if fact.id == id));
    lines.len() != before
}

/// A queued proposal whose source changed since it was proposed.
#[derive(Debug, Clone, Serialize)]
pub struct QueueProblem {
    pub model: String,
    pub queue_id: i64,
    pub message: String,
}

/// `localrag check` for brains: every brain file in the workspace, and every pending
/// proposal's citations.
pub fn check_all(ws: &Workspace, conn: &Connection) -> anyhow::Result<(Vec<Checked>, Vec<QueueProblem>)> {
    let mut models = file::models_with_brains(ws);
    let queued: Vec<String> = conn.prepare("SELECT DISTINCT model FROM fact_queue WHERE status = 'pending'")?.query_map([], |r| r.get(0))?.collect::<Result<_, _>>()?;
    let mut checked = Vec::new();
    for m in &models {
        checked.push(load_checked(ws, conn, m)?.1);
    }
    models.extend(queued);
    models.sort();
    models.dedup();
    let mut problems = Vec::new();
    for m in &models {
        for item in queue::pending(conn, m)? {
            let (_, status) = integrity::resolve_citations(conn, m, None, &item.citations)?;
            if status != "active" {
                problems.push(QueueProblem { model: m.clone(), queue_id: item.id, message: format!("proposal {} (“{}”): a cited source changed ({status}); it will be re-verified before approval", item.id, item.claim) });
            }
        }
    }
    Ok((checked, problems))
}
