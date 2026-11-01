//! `organize_brain` (§12): the model rearranges its brain (headings, `[THOUGHT]` lines, fact
//! order) and refers to facts only by ID; the program puts the original lines back byte for
//! byte, validates the result (§13), and keeps the previous version for Undo.

use std::collections::HashMap;

use serde::Serialize;
use serde_json::json;

use super::file::{self, Line};
use super::integrity::{Checked, LineView};
use crate::llm::{ChatRequest, Llm, Message};
use crate::prompts;
use crate::workspace::Workspace;

#[derive(Debug, Clone, Serialize)]
pub struct Organized {
    pub changed: bool,
    pub attempts: usize,
    /// Validation errors of the last failed attempt (when nothing was saved).
    pub errors: Vec<String>,
}

/// What the model is shown: the whole file, facts as `[FACT F-0012] claim (citation)`.
pub fn layout_for_model(checked: &Checked) -> String {
    let mut out = String::new();
    for l in &checked.lines {
        let s = match l {
            LineView::Heading { level, text } => format!("{} {text}", "#".repeat(*level)),
            LineView::Thought { text, .. } => format!("[THOUGHT] {text}"),
            LineView::Fact(f) if f.status != "unresolved" => {
                format!("[FACT {}] {} ({})", f.id, f.claim, f.citations.iter().map(|c| c.label.as_str()).collect::<Vec<_>>().join("; "))
            }
            LineView::Blank => String::new(),
            // Unresolved facts and invalid lines aren't the model's to place.
            _ => continue,
        };
        out.push_str(&s);
        out.push('\n');
    }
    out
}

/// Turns the model's layout into file lines, or the reasons it can't be used.
pub fn apply_layout(original: &[Line], layout: &[String]) -> Result<Vec<Line>, Vec<String>> {
    let facts: HashMap<&str, &Line> = original.iter().filter_map(|l| if let Line::Fact { fact, .. } = l { Some((fact.id.as_str(), l)) } else { None }).collect();
    // Facts with problems stay where the program can see them, after the layout.
    let mut errors = Vec::new();
    let mut out: Vec<Line> = Vec::new();
    let mut placed: HashMap<String, usize> = HashMap::new();
    let header = original.first().cloned();
    let mention = regex::Regex::new(r"\bF-\d{4,}\b").expect("valid regex");
    for (n, raw) in layout.iter().enumerate() {
        let t = raw.trim();
        if let Some(rest) = t.strip_prefix("[FACT ") {
            let id = rest.split(']').next().unwrap_or("").trim();
            match facts.get(id) {
                Some(l) => {
                    *placed.entry(id.to_string()).or_default() += 1;
                    out.push((*l).clone());
                }
                None => errors.push(format!("line {}: [FACT {id}] isn't a fact in this brain", n + 1)),
            }
            continue;
        }
        match file::parse_line(t) {
            Line::Fact { .. } => errors.push(format!("line {}: write facts as [FACT <id>] only; their text is filled in by the program", n + 1)),
            Line::Invalid { error, .. } => errors.push(format!("line {}: {error}", n + 1)),
            Line::Thought { text } if text.contains("[FACT") => errors.push(format!("line {}: a [THOUGHT] can't contain a [FACT] marker", n + 1)),
            Line::Thought { text } => {
                for m in mention.find_iter(&text) {
                    if !facts.contains_key(m.as_str()) {
                        errors.push(format!("line {}: the thought mentions {}, which isn't in this brain", n + 1, m.as_str()));
                    }
                }
                out.push(Line::Thought { text });
            }
            other => out.push(other),
        }
    }
    for id in facts.keys() {
        match placed.get(*id).copied().unwrap_or(0) {
            0 => errors.push(format!("[FACT {id}] is missing: every fact must appear exactly once")),
            1 => {}
            n => errors.push(format!("[FACT {id}] appears {n} times: every fact must appear exactly once")),
        }
    }
    // The header (with the model name) always stays first.
    if let Some(h @ Line::Heading { level: 1, .. }) = header
        && out.first() != Some(&h)
    {
        out.retain(|l| *l != h);
        out.insert(0, h);
    }
    // Final guard: exactly the same fact lines before and after.
    let mut before: Vec<String> = original.iter().filter(|l| matches!(l, Line::Fact { .. })).map(Line::raw).collect();
    let mut after: Vec<String> = out.iter().filter(|l| matches!(l, Line::Fact { .. })).map(Line::raw).collect();
    before.sort();
    after.sort();
    if errors.is_empty() && before != after {
        errors.push("the fact lines changed".into());
    }
    if errors.is_empty() { Ok(out) } else { Err(errors) }
}

/// Asks `model` to organize its brain; one retry with the errors; saves only a valid result.
pub fn organize(ws: &Workspace, llm: &dyn Llm, model: &str, num_ctx: u32, checked: &Checked) -> anyhow::Result<Organized> {
    if checked.lines.iter().any(|l| matches!(l, LineView::Fact(f) if f.status == "unresolved")) || checked.lines.iter().any(|l| matches!(l, LineView::Invalid { .. })) {
        anyhow::bail!("fix the integrity problems first (unresolved facts or invalid lines), so every fact can be placed");
    }
    let brain = file::load(ws, model)?;
    let layout = layout_for_model(checked);
    let schema = json!({ "type": "object", "properties": { "lines": { "type": "array", "items": { "type": "string" } } }, "required": ["lines"] });
    let mut messages = vec![Message::system(prompts::ORGANIZE_SYSTEM), Message::user(prompts::organize_request(&layout))];
    let mut errors = Vec::new();
    for attempt in 1..=2 {
        let req = ChatRequest { model: model.to_string(), messages: messages.clone(), num_ctx, temperature: 0.2, think: false, num_predict: None, format: Some(schema.clone()), tools: None };
        let reply = llm.chat(&req, &mut |_| {})?.content;
        let lines: Vec<String> = serde_json::from_str::<serde_json::Value>(reply.trim())
            .ok()
            .and_then(|v| v["lines"].as_array().map(|a| a.iter().filter_map(|l| l.as_str().map(str::to_string)).collect()))
            .unwrap_or_default();
        match apply_layout(&brain.lines, &lines) {
            Ok(new_lines) => {
                let changed = file::render(&new_lines) != brain.text;
                if changed {
                    file::update(ws, model, Some(&brain.hash), true, |l| {
                        *l = new_lines;
                        Ok(())
                    })?;
                }
                return Ok(Organized { changed, attempts: attempt, errors: Vec::new() });
            }
            Err(e) => {
                errors = e;
                messages.push(Message::assistant(reply));
                messages.push(Message::user(format!("That layout can't be used:\n- {}\nReturn the corrected layout.", errors.join("\n- "))));
            }
        }
    }
    Ok(Organized { changed: false, attempts: 2, errors })
}
