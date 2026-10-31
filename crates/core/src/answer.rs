//! `RagEngine::ask`: budget check → retrieval → gate → generation → citation validation
//! (ARCHITECTURE.md §10, §11).

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

use crate::budget::{self, ContextWarning, FixedCosts, PER_MESSAGE_OVERHEAD, TokenCounter};
use crate::citations::{self, Marker};
use crate::embed::Embedder;
use crate::llm::{ChatRequest, Delta, Llm, Message, ModelInfo};
use crate::prompts;
use crate::retrieval::{self, ChunkRef, Hit, SearchOptions};
use crate::workspace::Workspace;

#[derive(Debug, Clone)]
pub struct Turn {
    pub question: String,
    pub answer: String,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SpanSource {
    /// The model's quote was found in the source.
    Quote,
    /// No usable quote; the source sentence closest to the claim.
    Similar,
    None,
}

/// How far a citation is backed by its source (§10).
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum CitationStatus {
    /// The quote was found in the cited chunk.
    Verified,
    /// No quote, or the quote isn't in the chunk.
    #[default]
    Unverified,
    /// Unverified, then reviewed and accepted by the user.
    VerifiedParaphrase,
}

impl CitationStatus {
    pub fn counts_as_support(self) -> bool {
        matches!(self, CitationStatus::Verified | CitationStatus::VerifiedParaphrase)
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Citation {
    /// Position of this citation among all marker items in the answer, in order; the UI's
    /// placeholders (`citations::placeholders`) refer to it.
    pub ordinal: usize,
    pub status: CitationStatus,
    /// The `n` in `[Sn]`.
    pub marker: usize,
    pub chunk: ChunkRef,
    pub quote: Option<String>,
    pub quote_verified: bool,
    /// Char range in `chunk.text` to highlight.
    pub span: Option<(usize, usize)>,
    pub span_source: SpanSource,
}

#[derive(Debug, Clone, Serialize)]
pub struct Answer {
    /// Markdown with `[S#]` markers (rendered to `[Author Year, p. N]` for display).
    pub text: String,
    pub supported: bool,
    /// True when retrieval found nothing relevant enough and the model wasn't asked.
    pub not_found: bool,
    pub citations: Vec<Citation>,
    /// Markers naming a source number that wasn't given, e.g. `S12` with 8 sources.
    pub invalid_markers: Vec<String>,
    pub sources: Vec<ChunkRef>,
    pub unsupported_sentences: Vec<String>,
    pub suggestions: Vec<String>,
    pub warnings: Vec<ContextWarning>,
    pub model: String,
    pub thinking: Option<String>,
    /// The retrieval query (the question, or its standalone rewrite for follow-ups).
    pub query: String,
    pub num_ctx: u32,
}

impl Answer {
    /// Replaces `[S#…]` markers with `[Author Year, p. N]` labels.
    pub fn rendered(&self) -> String {
        let markers = citations::parse_markers(&self.text);
        let mut out = String::new();
        let mut i = 0;
        for m in &markers {
            out.push_str(&self.text[i..m.range.start]);
            let labels: Vec<String> = m
                .items
                .iter()
                .map(|it| self.sources.get(it.source - 1).map_or(format!("S{}?", it.source), |c| c.citation()))
                .collect();
            out.push_str(&format!("[{}]", labels.join("; ")));
            i = m.range.end;
        }
        out.push_str(&self.text[i..]);
        out
    }
}

pub enum Event<'a> {
    Query(&'a str),
    Sources(&'a [ChunkRef]),
    Warning(&'a ContextWarning),
    Thinking(&'a str),
    Token(&'a str),
}

pub struct RagEngine<'a> {
    pub ws: &'a Workspace,
    pub conn: &'a Connection,
    pub embedder: &'a dyn Embedder,
    pub llm: &'a dyn Llm,
}

const NOT_FOUND_TEXT: &str = "The indexed literature doesn't contain enough information to answer this question.";

impl RagEngine<'_> {
    /// Fails with a downcastable [`budget::OverBudget`] when the question can't fit.
    /// The submit check alone (§11): fails with a downcastable [`budget::OverBudget`].
    pub fn check_budget(&self, question: &str, model: &str, num_ctx: u32) -> anyhow::Result<()> {
        let info = self.llm.model_info(model)?;
        let counter = TokenCounter::for_model(self.conn, &info);
        self.fixed_costs(&info, &counter, question, num_ctx).check()?;
        Ok(())
    }

    fn fixed_costs(&self, info: &ModelInfo, counter: &TokenCounter, question: &str, num_ctx: u32) -> FixedCosts {
        let think = self.ws.config.think && info.can("thinking");
        FixedCosts {
            num_ctx,
            system: counter.count(prompts::SYSTEM),
            output_reserve: if think { budget::OUTPUT_RESERVE_THINKING } else { budget::OUTPUT_RESERVE },
            min_grounding: 3 * self.ws.config.chunking.target_tokens,
            question: counter.count(&prompts::question_with_sources(question, &[])),
        }
    }

    pub fn ask(&self, question: &str, model: &str, num_ctx: u32, history: &[Turn], on: &mut dyn FnMut(Event)) -> anyhow::Result<Answer> {
        let cfg = &self.ws.config;
        let info = self.llm.model_info(model)?;
        let counter = TokenCounter::for_model(self.conn, &info);
        let think = cfg.think && info.can("thinking");
        let fixed = self.fixed_costs(&info, &counter, question, num_ctx);
        let min_grounding = fixed.min_grounding;
        fixed.check()?;

        let mut answer = Answer {
            text: String::new(),
            supported: false,
            not_found: false,
            citations: Vec::new(),
            invalid_markers: Vec::new(),
            sources: Vec::new(),
            unsupported_sentences: Vec::new(),
            suggestions: Vec::new(),
            warnings: Vec::new(),
            model: model.to_string(),
            thinking: None,
            query: question.to_string(),
            num_ctx,
        };

        if !history.is_empty() {
            answer.query = self.standalone_query(model, num_ctx, history, question)?;
            on(Event::Query(&answer.query));
        }

        let opts = SearchOptions { candidates: cfg.retrieval.candidates, limit: cfg.retrieval.max_sources * 2, per_doc_cap: cfg.retrieval.per_doc_cap };
        let hits = retrieval::search(self.conn, self.embedder, &answer.query, &opts)?;
        let min_sim = cfg.retrieval.min_similarity;
        let best = hits.iter().filter_map(|h| h.similarity).fold(f32::MIN, f32::max);
        if hits.is_empty() || best < min_sim {
            answer.not_found = true;
            answer.text = NOT_FOUND_TEXT.to_string();
            answer.sources = hits.iter().take(3).map(|h| h.chunk.clone()).collect();
            on(Event::Sources(&answer.sources));
            answer.suggestions = self.suggestions(model, num_ctx, question, &answer.sources)?;
            return Ok(answer);
        }

        // History first (newest turns), leaving at least the grounding minimum for sources.
        let mut available = fixed.available_for_context();
        let mut history_budget = available.saturating_sub(min_grounding);
        let mut kept: Vec<&Turn> = Vec::new();
        let mut dropped_tokens = 0;
        for (i, turn) in history.iter().enumerate().rev() {
            let cost = counter.count(&turn.question) + counter.count(&turn.answer) + 2 * PER_MESSAGE_OVERHEAD;
            if cost <= history_budget && kept.len() == history.len() - 1 - i {
                history_budget -= cost;
                available -= cost;
                kept.push(turn);
            } else {
                dropped_tokens += cost;
            }
        }
        kept.reverse();
        if kept.len() < history.len() {
            self.warn(&mut answer, on, ContextWarning::HistoryDropped { turns: history.len() - kept.len(), tokens: dropped_tokens });
        }

        // Then sources, best first, until the budget or max_sources is reached.
        let relevant = |h: &Hit| h.similarity.is_some_and(|s| s >= min_sim) || h.bm25_rank.is_some_and(|r| r <= 5);
        let mut used = 0;
        let mut dropped_relevant = 0;
        let mut included_relevant = 0;
        for h in hits.iter() {
            if answer.sources.len() == cfg.retrieval.max_sources {
                break;
            }
            let cost = counter.count(&prompts::source_block(answer.sources.len() + 1, &h.chunk));
            if used + cost <= available {
                used += cost;
                included_relevant += relevant(h) as usize;
                answer.sources.push(h.chunk.clone());
            } else if relevant(h) {
                dropped_relevant += 1;
            }
        }
        if dropped_relevant > 0 {
            self.warn(&mut answer, on, ContextWarning::SourcesDropped { dropped: dropped_relevant, relevant: included_relevant + dropped_relevant });
        }

        // Neighbouring chunks of the top hits, as extra numbered sources.
        let mut neighbors_dropped = 0;
        let top: Vec<ChunkRef> = answer.sources.iter().take(cfg.retrieval.expand_top).cloned().collect();
        for chunk in &top {
            for n in retrieval::neighbors(self.conn, chunk)? {
                if answer.sources.iter().any(|s| s.id == n.id) {
                    continue;
                }
                let cost = counter.count(&prompts::source_block(answer.sources.len() + 1, &n));
                if used + cost <= available {
                    used += cost;
                    answer.sources.push(n);
                } else {
                    neighbors_dropped += 1;
                }
            }
        }
        if neighbors_dropped > 0 {
            self.warn(&mut answer, on, ContextWarning::NeighborsDropped { passages: neighbors_dropped });
        }
        on(Event::Sources(&answer.sources));

        let mut messages = vec![Message::system(prompts::SYSTEM)];
        for t in &kept {
            messages.push(Message::user(t.question.clone()));
            messages.push(Message::assistant(t.answer.clone()));
        }
        messages.push(Message::user(prompts::question_with_sources(question, &answer.sources)));
        let sent: usize = messages.iter().map(|m| counter.count(&m.content) + PER_MESSAGE_OVERHEAD).sum();

        let req = ChatRequest { model: model.to_string(), messages: messages.clone(), num_ctx, temperature: 0.2, think, num_predict: None };
        let resp = self.llm.chat(&req, &mut |d| match d {
            Delta::Thinking(t) => on(Event::Thinking(t)),
            Delta::Content(t) => on(Event::Token(t)),
        })?;
        if let Some(evaluated) = resp.prompt_eval_count {
            let near_limit = evaluated as usize + 8 >= num_ctx as usize;
            let short = (evaluated as f32) < sent as f32 * 0.85 && sent > 200;
            if near_limit || (counter.is_exact() && short) {
                self.warn(&mut answer, on, ContextWarning::Truncated { sent, evaluated });
            }
            if !counter.is_exact() && !near_limit {
                let chars: usize = messages.iter().map(|m| m.content.chars().count()).sum();
                budget::calibrate(self.conn, model, chars, evaluated)?;
            }
        }
        answer.thinking = Some(resp.thinking).filter(|t| !t.is_empty());
        answer.text = resp.content.trim().to_string();

        if answer.text.starts_with("INSUFFICIENT") {
            answer.suggestions = self.suggestions(model, num_ctx, question, &answer.sources[..answer.sources.len().min(3)])?;
            return Ok(answer);
        }
        let markers = citations::parse_markers(&answer.text);
        self.resolve_citations(&mut answer, &markers);
        answer.unsupported_sentences = citations::uncited_sentences(&answer.text, &markers);
        answer.supported = answer.citations.iter().any(|c| c.status.counts_as_support());
        Ok(answer)
    }

    fn resolve_citations(&self, answer: &mut Answer, markers: &[Marker]) {
        let mut ordinal = 0;
        for m in markers {
            let claim = claim_before(&answer.text, m.range.start);
            for item in &m.items {
                ordinal += 1;
                let Some(chunk) = answer.sources.get(item.source - 1).cloned() else {
                    answer.invalid_markers.push(format!("S{}", item.source));
                    continue;
                };
                let verified = item.quote.as_deref().and_then(|q| citations::verify_quote(q, &chunk.text, chunk.ocr));
                let (span, span_source) = match verified {
                    Some(r) => (Some((r.start, r.end)), SpanSource::Quote),
                    None => match citations::best_sentence(&claim, &chunk.text) {
                        Some(r) => (Some((r.start, r.end)), SpanSource::Similar),
                        None => (None, SpanSource::None),
                    },
                };
                answer.citations.push(Citation {
                    ordinal: ordinal - 1,
                    status: if span_source == SpanSource::Quote { CitationStatus::Verified } else { CitationStatus::Unverified },
                    marker: item.source,
                    quote: item.quote.clone(),
                    quote_verified: span_source == SpanSource::Quote,
                    span,
                    span_source,
                    chunk,
                });
            }
        }
    }

    fn warn(&self, answer: &mut Answer, on: &mut dyn FnMut(Event), w: ContextWarning) {
        on(Event::Warning(&w));
        answer.warnings.push(w);
    }

    fn standalone_query(&self, model: &str, num_ctx: u32, history: &[Turn], question: &str) -> anyhow::Result<String> {
        let pairs: Vec<(String, String)> = history.iter().map(|t| (t.question.clone(), t.answer.clone())).collect();
        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![Message::system(prompts::REWRITE_SYSTEM), Message::user(prompts::rewrite_request(&pairs, question))],
            num_ctx,
            temperature: 0.0,
            think: false,
            num_predict: Some(80),
        };
        let resp = self.llm.chat(&req, &mut |_| {})?;
        let q = resp.content.lines().find(|l| !l.trim().is_empty()).unwrap_or(question).trim().trim_matches('"');
        Ok(if q.is_empty() { question.to_string() } else { q.to_string() })
    }

    fn suggestions(&self, model: &str, num_ctx: u32, question: &str, closest: &[ChunkRef]) -> anyhow::Result<Vec<String>> {
        let req = ChatRequest {
            model: model.to_string(),
            messages: vec![Message::system(prompts::SUGGEST_SYSTEM), Message::user(prompts::suggest_request(question, closest))],
            num_ctx,
            temperature: 0.3,
            think: false,
            num_predict: Some(300),
        };
        let resp = self.llm.chat(&req, &mut |_| {})?;
        Ok(resp
            .content
            .lines()
            .map(|l| l.trim().trim_start_matches(['-', '*', '•']).trim())
            .map(|l| l.trim_start_matches(|c: char| c.is_ascii_digit()).trim_start_matches(['.', ')']).trim())
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect())
    }
}

/// The sentence a marker at byte `pos` supports: text since the previous sentence end, or
/// the sentence before when the marker follows a full stop.
fn claim_before(text: &str, pos: usize) -> String {
    let before = &text[..pos];
    let ends = |s: &str| s.rfind(['.', '!', '?', '\n']);
    match ends(before) {
        Some(i) if before[i + 1..].trim().is_empty() => {
            let prev = &before[..i];
            prev[ends(prev).map_or(0, |j| j + 1)..].trim().to_string()
        }
        Some(i) => before[i + 1..].trim().to_string(),
        None => before.trim().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::claim_before;

    #[test]
    fn claims() {
        let t = "First fact here. Second fact [S1]. Third fact. [S2]";
        assert_eq!(claim_before(t, t.find("[S1]").unwrap()), "Second fact");
        assert_eq!(claim_before(t, t.find("[S2]").unwrap()), "Third fact");
    }
}
