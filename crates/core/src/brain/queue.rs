//! The Fact queue (§12): proposals pass the admission rules before they're queued, and reach
//! the brain only when you approve them.

use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::file::{self, FactCitation, FactLine};
use super::integrity::{self, CitationView};
use super::registry;
use crate::answer::{Support, judge_support};
use crate::citations;
use crate::db;
use crate::embed::Embedder;
use crate::llm::{ChatRequest, Llm, Message};
use crate::prompts;
use crate::workspace::Workspace;

/// At or above this similarity to a rejected or deleted claim, a proposal is dropped.
pub const REJECTED_SIMILARITY: f32 = 0.92;
/// At or above this similarity to an active fact, a proposal becomes a merge.
pub const DUPLICATE_SIMILARITY: f32 = 0.92;
/// Active facts at least this similar are checked for contradictions.
pub const RELATED_SIMILARITY: f32 = 0.75;

/// Everything the admission rules and queue actions need.
pub struct Gate<'a> {
    pub ws: &'a Workspace,
    pub conn: &'a Connection,
    pub embedder: &'a dyn Embedder,
    pub llm: &'a dyn Llm,
    /// The brain's owner: the reasoning model (also asked the closed-book probe).
    pub model: &'a str,
    /// Judges support, knowledge and contradictions.
    pub judge_model: &'a str,
    pub num_ctx: u32,
}

#[derive(Debug, Clone)]
pub struct Proposal {
    pub claim: String,
    pub citations: Vec<FactCitation>,
    pub reason: String,
    /// The sub-question whose answer the citations came from (for the closed-book probe).
    pub question: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "outcome", rename_all = "snake_case")]
pub enum Admission {
    Queued { queue_id: i64 },
    /// A near-duplicate of an active fact: queued as adding its citations to that fact.
    Merge { queue_id: i64, target: String },
    Dropped { rule: String, reason: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Related {
    pub id: String,
    pub claim: String,
    /// conflicts | duplicate | related
    pub relation: String,
    pub similarity: f32,
}

fn dropped(rule: &str, reason: impl Into<String>) -> Admission {
    Admission::Dropped { rule: rule.into(), reason: reason.into() }
}

fn chunk_text(conn: &Connection, id: Option<i64>) -> anyhow::Result<Option<(String, bool)>> {
    let Some(id) = id else { return Ok(None) };
    Ok(conn.query_row("SELECT text, ocr FROM chunks WHERE id = ?1", [id], |r| Ok((r.get(0)?, r.get(1)?))).optional()?)
}

impl Gate<'_> {
    /// Rule 1: every quote is in its (current) source, and the strict check finds the claim
    /// supported: no passage says "no" and at least one says "yes".
    fn verify(&self, claim: &str, cites: &[FactCitation]) -> anyhow::Result<Result<Vec<CitationView>, String>> {
        if cites.is_empty() {
            return Ok(Err("it has no citations".into()));
        }
        let (views, status) = integrity::resolve_citations(self.conn, self.model, None, cites)?;
        if status != "active" {
            let bad = views.iter().find(|v| !matches!(v.resolution, crate::refs::Resolution::Resolved | crate::refs::Resolution::Relocated)).map(|v| v.label.clone()).unwrap_or_default();
            return Ok(Err(format!("the quote from {bad} isn't in the corpus")));
        }
        let mut items = Vec::new();
        for v in &views {
            let Some((text, ocr)) = chunk_text(self.conn, v.chunk_id)? else { return Ok(Err("a cited passage is gone".into())) };
            if citations::verify_quote(&v.quote, &text, ocr).is_none() {
                return Ok(Err(format!("the quote from {} doesn't verify", v.label)));
            }
            items.push((claim.to_string(), text));
        }
        let verdicts = judge_support(self.llm, self.judge_model, self.num_ctx, &items)?;
        if verdicts.contains(&Some(Support::No)) || !verdicts.contains(&Some(Support::Yes)) {
            return Ok(Err("the strict check didn't find the claim supported by its passages".into()));
        }
        Ok(Ok(views))
    }

    /// Rule 3: asks the brain's model the question with no sources. If its unaided answer
    /// already states the claim, the fact adds nothing for this model.
    fn already_known(&self, question: &str, claim: &str) -> anyhow::Result<bool> {
        let probe = ChatRequest {
            model: self.model.to_string(),
            messages: vec![Message::system(prompts::PROBE_SYSTEM), Message::user(question)],
            num_ctx: self.num_ctx.min(8192),
            temperature: 0.0,
            think: false,
            num_predict: Some(200),
            format: None,
            tools: None,
        };
        let unaided = self.llm.chat(&probe, &mut |_| {})?.content;
        let req = ChatRequest {
            model: self.judge_model.to_string(),
            messages: vec![Message::system(prompts::KNOWN_SYSTEM), Message::user(prompts::known_request(claim, &unaided))],
            num_ctx: self.num_ctx.min(8192),
            temperature: 0.0,
            think: false,
            num_predict: Some(30),
            format: Some(json!({ "type": "object", "properties": { "same": { "type": "boolean" } }, "required": ["same"] })),
            tools: None,
        };
        let v: serde_json::Value = serde_json::from_str(self.llm.chat(&req, &mut |_| {})?.content.trim()).unwrap_or_default();
        Ok(v["same"].as_bool() == Some(true))
    }

    /// Which of `others` contradict `claim` (rule 6).
    fn contradictions(&self, claim: &str, others: &[(String, String)]) -> anyhow::Result<Vec<bool>> {
        if others.is_empty() {
            return Ok(Vec::new());
        }
        let req = ChatRequest {
            model: self.judge_model.to_string(),
            messages: vec![Message::system(prompts::CONFLICT_SYSTEM), Message::user(prompts::conflict_request(claim, others))],
            num_ctx: self.num_ctx.min(8192),
            temperature: 0.0,
            think: false,
            num_predict: Some(30 * others.len() as i64 + 30),
            format: Some(json!({ "type": "object", "properties": { "verdicts": { "type": "array", "items": {
                "type": "object", "properties": { "item": { "type": "integer" }, "contradicts": { "type": "boolean" } }, "required": ["item", "contradicts"] } } }, "required": ["verdicts"] })),
            tools: None,
        };
        let v: serde_json::Value = serde_json::from_str(self.llm.chat(&req, &mut |_| {})?.content.trim()).unwrap_or_default();
        let mut out = vec![false; others.len()];
        for x in v["verdicts"].as_array().into_iter().flatten() {
            if let (Some(i), Some(c)) = (x["item"].as_u64(), x["contradicts"].as_bool())
                && let Some(slot) = out.get_mut(i as usize)
            {
                *slot = c;
            }
        }
        Ok(out)
    }

    /// The admission rules (§12), cheapest first; a proposal that passes is queued.
    pub fn propose(&self, p: &Proposal, run_id: Option<i64>) -> anyhow::Result<Admission> {
        let claim = p.claim.split_whitespace().collect::<Vec<_>>().join(" ");
        if claim.is_empty() {
            return Ok(dropped("cited", "the claim is empty"));
        }
        // 2. General and reusable: the agent must say why.
        if p.reason.trim().is_empty() {
            return Ok(dropped("reusable", "no reason was given for keeping it"));
        }
        // 4. Not previously rejected or deleted.
        if let Some((_, s)) = registry::nearest(self.conn, self.embedder, self.model, "rejected", &claim, 1)?.into_iter().next()
            && s >= REJECTED_SIMILARITY
        {
            return Ok(dropped("rejected_before", format!("a nearly identical claim was rejected or deleted before (similarity {s:.2})")));
        }
        // Already waiting in the queue.
        let pending: Vec<String> = self.conn.prepare("SELECT claim FROM fact_queue WHERE model = ?1 AND status = 'pending'")?.query_map([self.model], |r| r.get(0))?.collect::<Result<_, _>>()?;
        if pending.iter().any(|c| crate::metadata::title_similarity(c, &claim) >= 0.95) {
            return Ok(dropped("duplicate", "the same claim is already waiting in the queue"));
        }
        // 1. Cited and verified.
        if let Err(why) = self.verify(&claim, &p.citations)? {
            return Ok(dropped("cited", why));
        }
        // 3. Not already known to this model.
        if let Some(q) = p.question.as_deref().filter(|q| !q.trim().is_empty())
            && self.already_known(q, &claim)?
        {
            return Ok(dropped("already_known", "the model states this without sources; it adds nothing to its brain"));
        }
        // 5 and 6: duplicates become merges; contradictions are kept and linked.
        let near = registry::nearest(self.conn, self.embedder, self.model, "fact", &claim, 5)?;
        let mut related: Vec<Related> = Vec::new();
        for (id, s) in &near {
            let Some(f) = registry::get(self.conn, self.model, id)?.filter(|f| f.status != "deleted") else { continue };
            if *s >= RELATED_SIMILARITY {
                related.push(Related { id: id.clone(), claim: f.claim, relation: if *s >= DUPLICATE_SIMILARITY { "duplicate" } else { "related" }.into(), similarity: *s });
            }
        }
        let check: Vec<(String, String)> = related.iter().map(|r| (r.id.clone(), r.claim.clone())).collect();
        for (r, c) in related.iter_mut().zip(self.contradictions(&claim, &check)?) {
            if c {
                r.relation = "conflicts".into();
            }
        }
        let target = related.iter().find(|r| r.relation == "duplicate").map(|r| r.id.clone());
        self.conn.execute(
            "INSERT INTO fact_queue (model, kind, target_fact_id, claim, citations_json, reason, question, related_json, status, run_id, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'pending', ?9, ?10)",
            params![
                self.model,
                if target.is_some() { "merge" } else { "new" },
                target,
                claim,
                serde_json::to_string(&p.citations)?,
                p.reason.trim(),
                p.question,
                serde_json::to_string(&related)?,
                run_id,
                db::now_rfc3339()
            ],
        )?;
        let queue_id = self.conn.last_insert_rowid();
        Ok(match target {
            Some(target) => Admission::Merge { queue_id, target },
            None => Admission::Queued { queue_id },
        })
    }

    /// Approve (optionally with an edited claim, which is re-verified first). A merge item
    /// adds its citations to the target instead. Returns the fact ID.
    pub fn approve(&self, queue_id: i64, edited_claim: Option<&str>) -> anyhow::Result<String> {
        let item = get(self.conn, queue_id)?.filter(|i| i.model == self.model && i.status == "pending").ok_or_else(|| anyhow::anyhow!("no pending proposal {queue_id}"))?;
        if let (Some(target), None) = (&item.target_fact_id, edited_claim) {
            return self.merge(queue_id, target);
        }
        let claim = edited_claim.map(str::trim).filter(|c| !c.is_empty()).unwrap_or(&item.claim).to_string();
        if let Err(why) = self.verify(&claim, &item.citations)? {
            anyhow::bail!("can't approve: {why}");
        }
        let fact = registry::register(self.conn, self.model, &claim, &item.citations, item.reason.as_deref(), item.run_id)?;
        file::update(self.ws, self.model, None, false, |lines| {
            super::append_fact(lines, &fact);
            Ok(())
        })?;
        registry::embed(self.conn, self.embedder, self.model, "fact", &fact.id, &claim)?;
        for r in item.related.iter().filter(|r| r.relation == "conflicts") {
            registry::link_conflict(self.conn, self.model, &fact.id, &r.id)?;
        }
        self.decide(queue_id, "approved")?;
        Ok(fact.id)
    }

    /// Adds the proposal's citations to an active fact (its line is rewritten by the program).
    pub fn merge(&self, queue_id: i64, target: &str) -> anyhow::Result<String> {
        let item = get(self.conn, queue_id)?.filter(|i| i.model == self.model && i.status == "pending").ok_or_else(|| anyhow::anyhow!("no pending proposal {queue_id}"))?;
        let reg = registry::get(self.conn, self.model, target)?.filter(|f| f.status != "deleted").ok_or_else(|| anyhow::anyhow!("no active fact {target}"))?;
        let mut cites = reg.citations.clone();
        for c in &item.citations {
            if !cites.iter().any(|x| x.chunk_text_sha == c.chunk_text_sha && x.quote == c.quote) {
                cites.push(c.clone());
            }
        }
        if let Err(why) = self.verify(&reg.claim, &cites)? {
            anyhow::bail!("can't merge: {why}");
        }
        let fact = FactLine { id: reg.id.clone(), claim: reg.claim.clone(), citations: cites };
        registry::rewrite(self.conn, self.model, &fact)?;
        file::update(self.ws, self.model, None, false, |lines| {
            if !super::replace_fact(lines, &fact) {
                super::append_fact(lines, &fact);
            }
            Ok(())
        })?;
        self.decide(queue_id, "merged")?;
        Ok(fact.id)
    }

    pub fn reject(&self, queue_id: i64) -> anyhow::Result<()> {
        let item = get(self.conn, queue_id)?.filter(|i| i.model == self.model && i.status == "pending").ok_or_else(|| anyhow::anyhow!("no pending proposal {queue_id}"))?;
        registry::reject(self.conn, self.embedder, self.model, &item.claim, &item.citations, "queue_reject")?;
        self.decide(queue_id, "rejected")
    }

    fn decide(&self, queue_id: i64, status: &str) -> anyhow::Result<()> {
        self.conn.execute("UPDATE fact_queue SET status = ?2, decided_at = ?3 WHERE id = ?1", params![queue_id, status, db::now_rfc3339()])?;
        Ok(())
    }

    /// Edits an approved fact's claim: re-verified against its quotes, then the program
    /// rewrites the line. `expected_hash` guards against a concurrent edit of the file.
    pub fn edit_fact(&self, id: &str, claim: &str, expected_hash: Option<&str>) -> anyhow::Result<()> {
        let reg = registry::get(self.conn, self.model, id)?.filter(|f| f.status != "deleted").ok_or_else(|| anyhow::anyhow!("no fact {id}"))?;
        let claim = claim.split_whitespace().collect::<Vec<_>>().join(" ");
        if let Err(why) = self.verify(&claim, &reg.citations)? {
            anyhow::bail!("not saved: {why}");
        }
        let fact = FactLine { id: id.to_string(), claim: claim.clone(), citations: reg.citations };
        file::update(self.ws, self.model, expected_hash, false, |lines| {
            super::replace_fact(lines, &fact);
            Ok(())
        })?;
        registry::rewrite(self.conn, self.model, &fact)?;
        registry::embed(self.conn, self.embedder, self.model, "fact", id, &claim)
    }

    /// Deletes a fact: the line goes, the ID is tombstoned, and the claim is recorded as
    /// rejected so it isn't proposed again.
    pub fn delete_fact(&self, id: &str, expected_hash: Option<&str>) -> anyhow::Result<()> {
        file::update(self.ws, self.model, expected_hash, false, |lines| {
            super::remove_fact(lines, id);
            Ok(())
        })?;
        registry::delete(self.conn, self.embedder, self.model, id)
    }

    /// Puts a registered fact that went missing from the file back, byte for byte.
    pub fn restore_fact(&self, id: &str) -> anyhow::Result<()> {
        let reg = registry::get(self.conn, self.model, id)?.filter(|f| f.status != "deleted").ok_or_else(|| anyhow::anyhow!("no active fact {id}"))?;
        let file::Line::Fact { fact, .. } = file::parse_line(&reg.line) else { anyhow::bail!("the registered line of {id} doesn't parse") };
        file::update(self.ws, self.model, None, false, |lines| {
            if !lines.iter().any(|l| matches!(l, file::Line::Fact { fact: f, .. } if f.id == id)) {
                super::append_fact(lines, &fact);
            }
            Ok(())
        })?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct QueueItem {
    pub id: i64,
    pub model: String,
    pub kind: String,
    pub target_fact_id: Option<String>,
    pub claim: String,
    pub citations: Vec<FactCitation>,
    pub reason: Option<String>,
    pub question: Option<String>,
    pub related: Vec<Related>,
    pub status: String,
    pub run_id: Option<i64>,
    pub created_at: String,
}

type QueueRow = (i64, String, String, Option<String>, String, String, Option<String>, Option<String>, Option<String>, String, Option<i64>, String);

fn item(r: QueueRow) -> QueueItem {
    QueueItem {
        id: r.0,
        model: r.1,
        kind: r.2,
        target_fact_id: r.3,
        claim: r.4,
        citations: serde_json::from_str(&r.5).unwrap_or_default(),
        reason: r.6,
        question: r.7,
        related: r.8.as_deref().and_then(|j| serde_json::from_str(j).ok()).unwrap_or_default(),
        status: r.9,
        run_id: r.10,
        created_at: r.11,
    }
}

const COLUMNS: &str = "id, model, kind, target_fact_id, claim, citations_json, reason, question, related_json, status, run_id, created_at";

pub fn get(conn: &Connection, id: i64) -> anyhow::Result<Option<QueueItem>> {
    Ok(conn
        .query_row(&format!("SELECT {COLUMNS} FROM fact_queue WHERE id = ?1"), [id], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?, r.get(10)?, r.get(11)?))
        })
        .optional()?
        .map(item))
}

pub fn pending(conn: &Connection, model: &str) -> anyhow::Result<Vec<QueueItem>> {
    Ok(conn
        .prepare(&format!("SELECT {COLUMNS} FROM fact_queue WHERE model = ?1 AND status = 'pending' ORDER BY id"))?
        .query_map([model], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?, r.get(8)?, r.get(9)?, r.get(10)?, r.get(11)?)))?
        .map(|r| r.map(item))
        .collect::<Result<_, _>>()?)
}
