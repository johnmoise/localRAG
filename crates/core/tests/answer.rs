//! Grounded answers end to end with a scripted LLM (ARCHITECTURE.md §10, §11, §13).

use std::cell::RefCell;

use localrag_core::answer::{RagEngine, SpanSource, Turn};
use localrag_core::budget::{ContextWarning, OverBudget};
use localrag_core::config::{ChunkingConfig, GlobalConfig, WorkspaceConfig};
use localrag_core::embed::Embedder;
use localrag_core::index::{self, Parsers};
use localrag_core::llm::{ChatRequest, ChatResponse, Delta, Llm, ModelInfo};
use localrag_core::workspace::Workspace;
use localrag_core::{db, history, refs};
use rusqlite::Connection;

/// Bag-of-words vectors: texts sharing words are similar, unrelated texts are not.
struct WordEmbedder;

impl Embedder for WordEmbedder {
    fn model(&self) -> &str {
        "words"
    }
    fn embed_documents(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| bag(t)).collect())
    }
    fn embed_query(&self, q: &str) -> anyhow::Result<Vec<f32>> {
        Ok(bag(q))
    }
}

fn bag(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; 256];
    for w in text.split(|c: char| !c.is_alphanumeric()).filter(|w| w.len() > 3) {
        let h = w.to_lowercase().bytes().fold(7u32, |h, b| h.wrapping_mul(31).wrapping_add(b as u32));
        v[(h % 256) as usize] += 1.0;
    }
    v
}

/// Replies with `answer` to the main request and a fixed list to suggestion/rewrite prompts.
struct ScriptedLlm {
    answer: String,
    /// Reply to quote-repair requests.
    repair: String,
    /// Reply to strict-mode support checks.
    support: String,
    requests: RefCell<Vec<ChatRequest>>,
}

impl ScriptedLlm {
    fn new(answer: &str) -> Self {
        ScriptedLlm { answer: answer.into(), repair: "NONE".into(), support: "{}".into(), requests: RefCell::new(Vec::new()) }
    }
}

impl Llm for ScriptedLlm {
    fn chat(&self, req: &ChatRequest, on_delta: &mut dyn FnMut(Delta)) -> anyhow::Result<ChatResponse> {
        self.requests.borrow_mut().push(req.clone());
        let system = &req.messages[0].content;
        let content = if system.contains("Suggest 3 to 5") {
            "- Registry studies of second primary cancers\n- Survival statistics by country".to_string()
        } else if system.contains("standalone search query") {
            "quadratic attention cost".to_string()
        } else if system.contains("Copy, word for word") {
            self.repair.clone()
        } else if system.contains("decide whether the passage supports") {
            self.support.clone()
        } else {
            self.answer.clone()
        };
        on_delta(Delta::Content(&content));
        Ok(ChatResponse { content, prompt_eval_count: None, ..ChatResponse::default() })
    }
    fn model_info(&self, model: &str) -> anyhow::Result<ModelInfo> {
        Ok(ModelInfo { name: model.into(), capabilities: vec!["completion".into()], ..ModelInfo::default() })
    }
}

struct Fixture {
    _dir: tempfile::TempDir,
    ws: Workspace,
    conn: Connection,
}

impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        Workspace::init(dir.path()).unwrap();
        let mut config = WorkspaceConfig { chunking: ChunkingConfig { target_tokens: 60, overlap: 0.0 }, ..WorkspaceConfig::default() };
        config.retrieval.min_similarity = 0.3;
        let ws = Workspace { root: dir.path().canonicalize().unwrap(), config, global: GlobalConfig::default() };
        let conn = db::open(&ws.db_path()).unwrap();
        std::fs::write(
            ws.corpus_dir().join("attention.txt"),
            "Self-attention takes quadratic time in the sequence length of the input.\n\n\
             Linear attention variants reduce the cost of attention to linear time.\n",
        )
        .unwrap();
        std::fs::write(ws.corpus_dir().join("birds.txt"), "Arctic terns migrate between the polar regions every year.\n").unwrap();
        let mut f = Fixture { _dir: dir, ws, conn };
        f.sync();
        f
    }

    fn sync(&mut self) {
        index::sync(&self.ws, &mut self.conn, &WordEmbedder, &Parsers::new(&self.ws), &mut |_| {}).unwrap();
    }

    fn engine<'a>(&'a self, llm: &'a dyn Llm) -> RagEngine<'a> {
        RagEngine { ws: &self.ws, conn: &self.conn, embedder: &WordEmbedder, llm }
    }
}

const Q: &str = "How does the time cost of self-attention scale with sequence length?";

#[test]
fn verified_citations_uncited_sentences_and_invalid_markers() {
    let f = Fixture::new();
    let llm = ScriptedLlm::new(
        "Self-attention is quadratic in sequence length [S1 \"takes quadratic time in the sequence length\"]. \
         Linear variants are cheaper [S1 \"this quote is invented\"]. \
         This claim has no citation whatsoever in the answer. Another one cites a missing source [S9].",
    );
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    assert!(a.supported && !a.not_found);
    assert_eq!(a.citations.len(), 2);
    assert!(a.citations[0].quote_verified);
    assert_eq!(a.citations[0].span_source, SpanSource::Quote);
    let (s, e) = a.citations[0].span.unwrap();
    let span: String = a.citations[0].chunk.text.chars().skip(s).take(e - s).collect();
    assert_eq!(span, "takes quadratic time in the sequence length");
    assert!(!a.citations[1].quote_verified, "invented quote is flagged");
    assert_eq!(a.invalid_markers, ["S9"]);
    assert_eq!(a.unsupported_sentences, ["This claim has no citation whatsoever in the answer."]);
    assert!(a.rendered().contains("[attention, ll. 1"), "{}", a.rendered());
    assert!(a.warnings.is_empty(), "{:?}", a.warnings);
    // The model saw the sources and the question.
    let reqs = llm.requests.borrow();
    assert!(reqs[0].messages.last().unwrap().content.contains("[S1] attention"));
    assert_eq!(reqs[0].num_ctx, 16_384);
}

#[test]
fn unverifiable_answers_are_not_supported() {
    let f = Fixture::new();
    let llm = ScriptedLlm::new("Attention is quadratic [S1 \"words that are not in the source\"].");
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    assert!(!a.supported);
}

#[test]
fn off_topic_questions_skip_generation() {
    let f = Fixture::new();
    let llm = ScriptedLlm::new("should not be used");
    let a = f.engine(&llm).ask("What is the capital city of Australia?", "m", 16_384, &[], &mut |_| {}).unwrap();
    assert!(a.not_found && !a.supported);
    assert_eq!(a.suggestions.len(), 2);
    let reqs = llm.requests.borrow();
    assert_eq!(reqs.len(), 1, "only the suggestions call");
    assert!(reqs[0].messages[0].content.contains("Suggest 3 to 5"));

    // Reopened later, the answer still shows its suggestions and "not found" state.
    let session = history::create_session(&f.conn, "q", "m").unwrap();
    let id = history::record(&f.conn, session, "q", &a).unwrap();
    let m = history::message(&f.conn, id).unwrap();
    assert_eq!(m.extra["not_found"], true);
    assert_eq!(m.extra["suggestions"].as_array().unwrap().len(), 2);
}

#[test]
fn insufficient_replies_get_suggestions() {
    let f = Fixture::new();
    let llm = ScriptedLlm::new("INSUFFICIENT: the sources do not give memory costs.");
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    assert!(!a.supported && !a.not_found);
    assert!(a.text.starts_with("INSUFFICIENT"));
    assert!(!a.suggestions.is_empty());
}

#[test]
fn over_budget_questions_are_blocked_before_any_model_call() {
    let f = Fixture::new();
    let llm = ScriptedLlm::new("unused");
    let long = format!("{Q} {}", "Please consider this context carefully. ".repeat(400));
    let err = f.engine(&llm).ask(&long, "m", 4_096, &[], &mut |_| {}).unwrap_err();
    let over = err.downcast_ref::<OverBudget>().expect("OverBudget");
    assert!(over.over_by > 0);
    assert!(err.to_string().starts_with(&format!("Over token count by {} tokens", over.over_by)));
    assert!(llm.requests.borrow().is_empty());
}

#[test]
fn dropped_history_raises_a_warning() {
    let f = Fixture::new();
    let llm = ScriptedLlm::new("Quadratic [S1 \"takes quadratic time\"].");
    let long_answer = "A long earlier answer about attention. ".repeat(120);
    let history: Vec<Turn> = (0..6).map(|i| Turn { question: format!("Earlier question {i}?"), answer: long_answer.clone() }).collect();
    let a = f.engine(&llm).ask(Q, "m", 4_096, &history, &mut |_| {}).unwrap();
    let dropped = a.warnings.iter().find_map(|w| match w {
        ContextWarning::HistoryDropped { turns, .. } => Some(*turns),
        _ => None,
    });
    assert!(dropped.is_some_and(|n| n > 0 && n < 6), "{:?}", a.warnings);
    assert_eq!(a.query, "quadratic attention cost", "follow-ups are rewritten for retrieval");
    // The newest turns were kept, oldest dropped.
    let msgs = &llm.requests.borrow()[1].messages;
    assert!(msgs.iter().any(|m| m.content == "Earlier question 5?"));
    assert!(!msgs.iter().any(|m| m.content == "Earlier question 0?"));
}

#[test]
fn history_citations_resolve_then_track_removal_and_forgetting() {
    let mut f = Fixture::new();
    let llm = ScriptedLlm::new("Quadratic [S1 \"takes quadratic time\"].");
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    let session = history::create_session(&f.conn, Q, "m").unwrap();
    history::record(&f.conn, session, Q, &a).unwrap();
    assert_eq!(history::turns(&f.conn, session).unwrap().len(), 1);

    let r = refs::check_history(&f.conn).unwrap();
    assert_eq!((r.checked, r.resolved), (1, 1));

    let path = f.ws.corpus_dir().join("attention.txt");
    let kept = f.ws.root.join("attention.txt");
    std::fs::rename(&path, &kept).unwrap();
    f.sync();
    let r = refs::check_history(&f.conn).unwrap();
    assert_eq!((r.source_removed, r.broken.len()), (1, 0));
    assert!(r.ok());

    let id = index::find_document(&f.conn, "attention.txt").unwrap().unwrap();
    index::forget(&f.ws, &f.conn, id).unwrap();
    let r = refs::check_history(&f.conn).unwrap();
    assert_eq!(r.broken.len(), 1);
    assert!(!r.ok());
}

#[test]
fn search_filters_by_document_and_year() {
    use localrag_core::metadata::{self, crossref::{Author, Work}};
    use localrag_core::retrieval::{self, Filter, SearchOptions};
    let f = Fixture::new();
    let q = "attention migrate polar regions";
    let all = retrieval::search(&f.conn, &WordEmbedder, q, &SearchOptions::default()).unwrap();
    let docs = |hits: &[retrieval::Hit]| hits.iter().map(|h| h.chunk.doc_path.clone()).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(docs(&all).len(), 2);

    let birds = index::find_document(&f.conn, "birds.txt").unwrap().unwrap();
    let only = SearchOptions { filter: Filter { doc_ids: Some(vec![birds]), ..Filter::default() }, ..SearchOptions::default() };
    assert_eq!(docs(&retrieval::search(&f.conn, &WordEmbedder, q, &only).unwrap()), ["birds.txt".to_string()].into());

    let work = Work { doi: "10.1/b".into(), title: Some("Terns".into()), year: Some(2015), author: vec![Author { family: Some("Avis".into()), given: None, literal: None }], container: None, page: None, kind: None };
    metadata::apply_match(&f.conn, birds, &work, "crossref_doi").unwrap();
    let years = |from, to| SearchOptions { filter: Filter { year_from: from, year_to: to, ..Filter::default() }, ..SearchOptions::default() };
    assert_eq!(docs(&retrieval::search(&f.conn, &WordEmbedder, q, &years(Some(2010), Some(2020))).unwrap()), ["birds.txt".to_string()].into());
    assert!(retrieval::search(&f.conn, &WordEmbedder, q, &years(Some(2016), None)).unwrap().is_empty(), "no year → excluded; 2015 < 2016");
}

#[test]
fn unverified_quotes_are_repaired_from_the_source_when_possible() {
    let f = Fixture::new();
    let mut llm = ScriptedLlm::new("Attention is quadratic [S1 \"attention needs quadratic compute\"].");
    llm.repair = "\"takes quadratic time in the sequence length\"".into();
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    let c = &a.citations[0];
    assert!(c.quote_verified && c.quote_repaired, "{c:?}");
    assert_eq!(c.quote.as_deref(), Some("takes quadratic time in the sequence length"));
    assert!(a.supported);

    // A "repair" that isn't actually in the passage changes nothing.
    let mut llm = ScriptedLlm::new("Attention is quadratic [S1 \"attention needs quadratic compute\"].");
    llm.repair = "attention needs quadratic compute".into();
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    assert!(!a.citations[0].quote_verified && !a.citations[0].quote_repaired);
    assert!(!a.supported);
}

#[test]
fn strict_mode_rejects_citations_whose_passage_does_not_support_the_claim() {
    let mut f = Fixture::new();
    f.ws.config.strict = true;
    let mut llm = ScriptedLlm::new("Attention is linear [S1 \"takes quadratic time\"].");
    llm.support = r#"{"verdicts":[{"item":0,"verdict":"no"}]}"#.into();
    let a = f.engine(&llm).ask(Q, "m", 16_384, &[], &mut |_| {}).unwrap();
    assert!(a.citations[0].quote_verified, "the quote exists…");
    assert_eq!(a.citations[0].support, Some(localrag_core::answer::Support::No), "…but doesn't support the claim");
    assert!(!a.supported);
    let req = llm.requests.borrow().iter().find(|r| r.format.is_some()).cloned().expect("structured request");
    assert!(req.messages[1].content.contains("Claim: Attention is linear"));
}
