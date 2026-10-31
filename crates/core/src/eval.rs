//! `localrag eval`: retrieval recall and grounding quality on a per-workspace gold set
//! (ARCHITECTURE.md §17).
//!
//! `.localrag/eval.jsonl`, one object per line:
//! `{"question": "...", "expected_doc": "part of the file path", "expected_pages": ["1997"], "answerable": true}`

use std::path::Path;

use anyhow::Context;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::answer::{Answer, RagEngine};
use crate::embed::Embedder;
use crate::retrieval::{self, SearchOptions};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EvalItem {
    pub question: String,
    #[serde(default)]
    pub expected_doc: Option<String>,
    #[serde(default)]
    pub expected_pages: Vec<String>,
    #[serde(default = "yes")]
    pub answerable: bool,
}

fn yes() -> bool {
    true
}

pub fn load(path: &Path) -> anyhow::Result<Vec<EvalItem>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty() && !l.trim_start().starts_with("//"))
        .map(|(i, l)| serde_json::from_str(l).with_context(|| format!("{}:{}", path.display(), i + 1)))
        .collect()
}

#[derive(Debug, Clone, Serialize)]
pub struct RetrievalResult {
    pub question: String,
    pub answerable: bool,
    /// 1-based rank of the first hit from the expected document.
    pub doc_rank: Option<usize>,
    /// 1-based rank of the first hit from the expected document covering an expected page.
    pub page_rank: Option<usize>,
    pub best_similarity: Option<f32>,
}

pub fn retrieval(conn: &Connection, embedder: &dyn Embedder, items: &[EvalItem]) -> anyhow::Result<Vec<RetrievalResult>> {
    let opts = SearchOptions { limit: 10, ..SearchOptions::default() };
    let mut out = Vec::new();
    for item in items {
        let hits = retrieval::search(conn, embedder, &item.question, &opts)?;
        let want = item.expected_doc.as_deref().unwrap_or("\u{0}").to_lowercase();
        let mut doc_rank = None;
        let mut page_rank = None;
        for (i, h) in hits.iter().enumerate() {
            if !h.chunk.doc_path.to_lowercase().contains(&want) {
                continue;
            }
            doc_rank.get_or_insert(i + 1);
            if page_rank.is_none() && covers_page(conn, h.chunk.id, &item.expected_pages)? {
                page_rank = Some(i + 1);
            }
        }
        out.push(RetrievalResult {
            question: item.question.clone(),
            answerable: item.answerable,
            doc_rank,
            page_rank,
            best_similarity: hits.iter().filter_map(|h| h.similarity).reduce(f32::max),
        });
    }
    Ok(out)
}

/// Whether a chunk's printed-page (or PDF-page) range includes any expected page.
fn covers_page(conn: &Connection, chunk_id: i64, pages: &[String]) -> anyhow::Result<bool> {
    if pages.is_empty() {
        return Ok(false);
    }
    let (ls, le, ps, pe): (Option<String>, Option<String>, Option<i64>, Option<i64>) = conn.query_row(
        "SELECT page_label_start, page_label_end, pdf_page_start, pdf_page_end FROM chunks WHERE id = ?1",
        [chunk_id],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
    )?;
    let range = |a: Option<i64>, b: Option<i64>| a.zip(b.or(a));
    let labels = range(ls.and_then(|s| s.parse().ok()), le.and_then(|s| s.parse().ok()));
    let pdf = range(ps, pe);
    Ok(pages.iter().filter_map(|p| p.parse::<i64>().ok()).any(|p| {
        labels.is_some_and(|(a, b)| (a..=b).contains(&p)) || (labels.is_none() && pdf.is_some_and(|(a, b)| (a..=b).contains(&p)))
    }))
}

#[derive(Debug, Clone, Serialize)]
pub struct AnswerResult {
    pub question: String,
    pub answerable: bool,
    pub supported: bool,
    pub not_found: bool,
    pub citations: usize,
    pub verified: usize,
    /// Citations pointing into the expected document.
    pub on_expected_doc: usize,
    pub invalid_markers: usize,
    pub uncited_sentences: usize,
    pub warnings: usize,
    pub answer: String,
}

pub fn answer_result(item: &EvalItem, a: &Answer) -> AnswerResult {
    let want = item.expected_doc.as_deref().map(str::to_lowercase);
    AnswerResult {
        question: item.question.clone(),
        answerable: item.answerable,
        supported: a.supported,
        not_found: a.not_found,
        citations: a.citations.len(),
        verified: a.citations.iter().filter(|c| c.quote_verified).count(),
        on_expected_doc: want.map_or(0, |w| a.citations.iter().filter(|c| c.chunk.doc_path.to_lowercase().contains(&w)).count()),
        invalid_markers: a.invalid_markers.len(),
        uncited_sentences: a.unsupported_sentences.len(),
        warnings: a.warnings.len(),
        answer: a.rendered(),
    }
}

pub fn answers(engine: &RagEngine, items: &[EvalItem], model: &str, num_ctx: u32, progress: &mut dyn FnMut(usize, &AnswerResult)) -> anyhow::Result<Vec<AnswerResult>> {
    let mut out = Vec::new();
    for (i, item) in items.iter().enumerate() {
        let a = engine.ask(&item.question, model, num_ctx, &[], &mut |_| {})?;
        let r = answer_result(item, &a);
        progress(i, &r);
        out.push(r);
    }
    Ok(out)
}

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub retrieval_questions: usize,
    /// For calibrating `retrieval.min_similarity`: it should sit between these two.
    pub lowest_answerable_similarity: Option<f32>,
    pub highest_unanswerable_similarity: Option<f32>,
    pub doc_recall_at_5: f32,
    pub doc_recall_at_10: f32,
    pub page_recall_at_5: Option<f32>,
    pub page_recall_at_10: Option<f32>,
    pub answerable: usize,
    pub supported_rate: Option<f32>,
    pub unanswerable: usize,
    pub refusal_rate: Option<f32>,
    pub quote_verification_rate: Option<f32>,
    pub citation_precision: Option<f32>,
}

pub fn summarize(items: &[EvalItem], retrieval: &[RetrievalResult], answers: &[AnswerResult]) -> Summary {
    let rate = |n: usize, d: usize| (d > 0).then(|| n as f32 / d as f32);
    let unanswerable_sims: Vec<f32> = retrieval.iter().filter(|r| !r.answerable).filter_map(|r| r.best_similarity).collect();
    let answerable_sims: Vec<f32> = retrieval.iter().filter(|r| r.answerable).filter_map(|r| r.best_similarity).collect();
    let retrieval: Vec<&RetrievalResult> = retrieval
        .iter()
        .filter(|r| r.answerable && items.iter().any(|i| i.question == r.question && i.expected_doc.is_some()))
        .collect();
    let at = |ranks: &mut dyn Iterator<Item = Option<usize>>, k: usize| ranks.filter(|r| r.is_some_and(|r| r <= k)).count();
    let with_pages: Vec<&&RetrievalResult> = retrieval
        .iter()
        .filter(|r| items.iter().any(|i| i.question == r.question && !i.expected_pages.is_empty()))
        .collect();
    let answerable: Vec<&AnswerResult> = answers.iter().filter(|a| a.answerable).collect();
    let unanswerable: Vec<&AnswerResult> = answers.iter().filter(|a| !a.answerable).collect();
    let cites: usize = answerable.iter().map(|a| a.citations).sum();
    Summary {
        retrieval_questions: retrieval.len(),
        lowest_answerable_similarity: answerable_sims.iter().copied().reduce(f32::min),
        highest_unanswerable_similarity: unanswerable_sims.iter().copied().reduce(f32::max),
        doc_recall_at_5: rate(at(&mut retrieval.iter().map(|r| r.doc_rank), 5), retrieval.len()).unwrap_or(0.0),
        doc_recall_at_10: rate(at(&mut retrieval.iter().map(|r| r.doc_rank), 10), retrieval.len()).unwrap_or(0.0),
        page_recall_at_5: rate(at(&mut with_pages.iter().map(|r| r.page_rank), 5), with_pages.len()),
        page_recall_at_10: rate(at(&mut with_pages.iter().map(|r| r.page_rank), 10), with_pages.len()),
        answerable: answerable.len(),
        supported_rate: rate(answerable.iter().filter(|a| a.supported).count(), answerable.len()),
        unanswerable: unanswerable.len(),
        refusal_rate: rate(unanswerable.iter().filter(|a| !a.supported).count(), unanswerable.len()),
        quote_verification_rate: rate(answerable.iter().map(|a| a.verified).sum(), cites),
        citation_precision: rate(answerable.iter().map(|a| a.on_expected_doc).sum(), cites),
    }
}
