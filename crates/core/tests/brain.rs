//! Second brain and reasoning agent end to end with a scripted LLM (ARCHITECTURE.md §12, §13).

use std::cell::RefCell;

use localrag_core::agent::Agent;
use localrag_core::brain::file::{self, Line};
use localrag_core::brain::queue::{Admission, Gate, Proposal};
use localrag_core::brain::{self, integrity, organize, registry};
use localrag_core::config::{ChunkingConfig, GlobalConfig, WorkspaceConfig};
use localrag_core::db;
use localrag_core::embed::Embedder;
use localrag_core::index::{self, Parsers};
use localrag_core::llm::{ChatRequest, ChatResponse, Delta, Llm, ModelInfo, ToolCall, ToolFunction};
use localrag_core::workspace::Workspace;
use rusqlite::Connection;
use serde_json::{Value, json};

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

const MODEL: &str = "thinker:1b";
const CLAIM: &str = "Self-attention takes quadratic time in the sequence length.";

/// The agent's moves, in order (one per model turn), then its final answer.
struct Scripted {
    moves: Vec<(&'static str, Value)>,
    final_text: String,
    organize: Value,
    /// The closed-book probe's answer.
    unaided: String,
    requests: RefCell<Vec<ChatRequest>>,
}

impl Scripted {
    fn new(moves: Vec<(&'static str, Value)>, final_text: &str) -> Self {
        Scripted { moves, final_text: final_text.into(), organize: json!({ "lines": [] }), unaided: "I don't know.".into(), requests: RefCell::new(Vec::new()) }
    }
}

impl Llm for Scripted {
    fn chat(&self, req: &ChatRequest, on: &mut dyn FnMut(Delta)) -> anyhow::Result<ChatResponse> {
        self.requests.borrow_mut().push(req.clone());
        let system = &req.messages[0].content;
        let mut resp = ChatResponse::default();
        if system.contains("careful research assistant") {
            let done = req.messages.iter().filter(|m| m.role == "tool").count();
            match self.moves.get(done).filter(|_| req.tools.is_some()) {
                Some((name, args)) => resp.tool_calls = vec![ToolCall { function: ToolFunction { name: name.to_string(), arguments: args.clone() } }],
                None => resp.content = self.final_text.clone(),
            }
        } else if system.contains("decide whether the passage supports") {
            resp.content = json!({ "verdicts": [{ "item": 0, "verdict": "yes" }, { "item": 1, "verdict": "yes" }] }).to_string();
        } else if system.contains("from your own knowledge only") {
            resp.content = self.unaided.clone();
        } else if system.contains("You compare a claim") {
            let same = req.messages[1].content.contains("quadratic time in the sequence length") && self.unaided.contains("quadratic");
            resp.content = json!({ "same": same }).to_string();
        } else if system.contains("contradicts the new statement") {
            resp.content = json!({ "verdicts": [{ "item": 0, "contradicts": req.messages[1].content.contains("linear time") }] }).to_string();
        } else if system.contains("You organize your own research notes") {
            resp.content = self.organize.to_string();
        } else if system.contains("Suggest 3 to 5") || system.contains("NOT in the collection") {
            resp.content = "- more papers".into();
        } else {
            // The grounded RAG answer to an ask_corpus sub-question.
            resp.content = "Self-attention is quadratic in sequence length [S1 \"takes quadratic time in the sequence length\"].".into();
        }
        on(Delta::Content(&resp.content));
        Ok(resp)
    }
    fn model_info(&self, model: &str) -> anyhow::Result<ModelInfo> {
        Ok(ModelInfo { name: model.into(), capabilities: vec!["completion".into(), "tools".into(), "thinking".into()], ..ModelInfo::default() })
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
        std::fs::write(ws.corpus_dir().join("attention.txt"), "Self-attention takes quadratic time in the sequence length of the input.\n").unwrap();
        std::fs::write(ws.corpus_dir().join("linear.txt"), "Linear attention variants reduce the cost of attention to linear time.\n").unwrap();
        let mut f = Fixture { _dir: dir, ws, conn };
        f.sync();
        f
    }

    fn sync(&mut self) {
        index::sync(&self.ws, &mut self.conn, &WordEmbedder, &Parsers::new(&self.ws), &mut |_| {}).unwrap();
    }

    fn agent<'a>(&'a self, llm: &'a dyn Llm) -> Agent<'a> {
        Agent { ws: &self.ws, conn: &self.conn, embedder: &WordEmbedder, llm, model: MODEL, num_ctx: 16_384, corpus_model: MODEL, corpus_num_ctx: 16_384 }
    }

    fn gate<'a>(&'a self, llm: &'a dyn Llm) -> Gate<'a> {
        Gate { ws: &self.ws, conn: &self.conn, embedder: &WordEmbedder, llm, model: MODEL, judge_model: MODEL, num_ctx: 16_384 }
    }

    fn checked(&self) -> integrity::Checked {
        brain::load_checked(&self.ws, &self.conn, MODEL).unwrap().1
    }

    fn file_text(&self) -> String {
        std::fs::read_to_string(file::path_for(&self.ws, MODEL)).unwrap()
    }
}

fn research_moves() -> Vec<(&'static str, Value)> {
    vec![
        ("read_brain", json!({ "query": "attention cost" })),
        ("ask_corpus", json!({ "question": "How does self-attention time scale with sequence length?" })),
        ("propose_fact", json!({ "claim": CLAIM, "citations": ["C1"], "reason": "A general cost property of attention." })),
        ("propose_fact", json!({ "claim": "Something invented.", "citations": ["C9"], "reason": "made up" })),
    ]
}

#[test]
fn an_agent_run_proposes_facts_and_approval_puts_them_in_the_right_brain() {
    let f = Fixture::new();
    let llm = Scripted::new(research_moves(), "Self-attention is quadratic [C1]. Something else [C7].");
    let mut events = Vec::new();
    let r = f.agent(&llm).run("Why is attention expensive on long inputs?", &mut |e| {
        if let localrag_core::agent::AgentEvent::ToolCall { name, .. } = e {
            events.push(name.to_string());
        }
    }).unwrap();
    assert_eq!(events, ["read_brain", "ask_corpus", "propose_fact", "propose_fact"]);
    assert!(r.text.starts_with("Self-attention is quadratic [attention"), "C1 is rendered as its source: {}", r.text);
    assert_eq!(r.invalid_markers, ["C7"], "a citation the agent was never given is flagged");
    assert_eq!(r.sources[0].quote.as_deref(), Some("takes quadratic time in the sequence length"));
    assert_eq!(r.proposals[0]["admission"]["outcome"], "queued");
    assert_eq!(r.proposals[1]["admission"]["outcome"], "dropped", "unknown citation ID C9");
    // The ask_corpus result showed the agent its citation IDs.
    let reqs = llm.requests.borrow();
    let tool_msg = reqs.iter().rev().find_map(|q| q.messages.iter().find(|m| m.role == "tool" && m.content.contains("Citations:"))).unwrap();
    assert!(tool_msg.content.contains("[C1] attention"), "{}", tool_msg.content);
    drop(reqs);

    // Queued facts aren't in the brain yet.
    let pending = brain::queue::pending(&f.conn, MODEL).unwrap();
    assert_eq!(pending.len(), 1);
    assert!(!file::path_for(&f.ws, MODEL).exists());
    assert!(brain::queue::pending(&f.conn, "other:7b").unwrap().is_empty(), "queues are per model");

    let id = f.gate(&llm).approve(pending[0].id, None).unwrap();
    assert_eq!(id, "F-0001");
    let text = f.file_text();
    assert!(text.starts_with("# Brain · thinker:1b\n"), "{text}");
    assert!(text.contains("## Unsorted\n[FACT] F-0001 | Self-attention takes quadratic time in the sequence length. | attention"), "{text}");
    let checked = f.checked();
    assert!(checked.ok(), "{:?}", checked.problems);
    let read = brain::read(&f.conn, &WordEmbedder, &checked, "attention", 4000).unwrap();
    assert!(read.text.contains("[F1] Self-attention takes quadratic time in the sequence length. (attention"), "{}", read.text);
    assert!(!read.text.contains("<doc:"), "hashes and quotes are dropped from the compact form");
    assert!(std::fs::read_to_string(f.ws.root.join("brain/.gitignore")).unwrap().contains('*'), "brain/ is kept out of git");

    // A second run reads F1 and may cite it; the answer shows F1's original source.
    let llm2 = Scripted::new(vec![("read_brain", json!({ "query": "attention" }))], "It is quadratic [F1].");
    let r2 = f.agent(&llm2).run("Is attention quadratic?", &mut |_| {}).unwrap();
    assert!(r2.invalid_markers.is_empty());
    assert_eq!(r2.sources[0].via, "F1");
    assert!(r2.text.contains("[attention"), "{}", r2.text);
    assert!(llm2.requests.borrow().iter().any(|q| q.messages.iter().any(|m| m.role == "tool" && m.content.contains("[F1]"))));
}

fn cite(f: &Fixture, needle: &str, quote: &str) -> file::FactCitation {
    let (sha, text_sha, label): (String, String, String) = f
        .conn
        .query_row(
            "SELECT d.sha256, c.text_sha, d.cite_label FROM chunks c JOIN documents d ON d.id = c.doc_id WHERE c.text LIKE ?1",
            [format!("%{needle}%")],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();
    file::FactCitation { label, quote: quote.into(), doc_sha256: sha, chunk_text_sha: text_sha }
}

fn propose(f: &Fixture, llm: &Scripted, claim: &str, c: file::FactCitation, question: Option<&str>) -> Admission {
    f.gate(llm).propose(&Proposal { claim: claim.into(), citations: vec![c], reason: "general".into(), question: question.map(str::to_string) }, None).unwrap()
}

#[test]
fn admission_rules_drop_known_rejected_and_unverified_claims_and_merge_duplicates() {
    let f = Fixture::new();
    let llm = Scripted::new(vec![], "");
    let quad = cite(&f, "quadratic", "takes quadratic time in the sequence length");

    // 1. The quote must be in the source.
    let mut bad = quad.clone();
    bad.quote = "takes cubic time".into();
    assert!(matches!(propose(&f, &llm, CLAIM, bad, None), Admission::Dropped { rule, .. } if rule == "cited"));

    // 3. Already known to this model without sources.
    let mut knows = Scripted::new(vec![], "");
    knows.unaided = "Self-attention is quadratic: it takes quadratic time in the sequence length.".into();
    assert!(matches!(propose(&f, &knows, CLAIM, quad.clone(), Some("How does attention scale?")), Admission::Dropped { rule, .. } if rule == "already_known"));

    // Queued, approved; then the same claim again is a merge into F-0001.
    let Admission::Queued { queue_id } = propose(&f, &llm, CLAIM, quad.clone(), Some("How does attention scale?")) else { panic!() };
    f.gate(&llm).approve(queue_id, None).unwrap();
    let again = propose(&f, &llm, "Self-attention takes quadratic time in sequence length.", quad.clone(), None);
    assert!(matches!(&again, Admission::Merge { target, .. } if target == "F-0001"), "{again:?}");

    // 6. A contradicting claim is kept and linked as a conflict on approval.
    let lin = cite(&f, "linear time", "reduce the cost of attention to linear time");
    let Admission::Queued { queue_id } = propose(&f, &llm, "Self-attention takes linear time in the sequence length.", lin.clone(), None) else { panic!("queued") };
    let item = brain::queue::get(&f.conn, queue_id).unwrap().unwrap();
    assert!(item.related.iter().any(|r| r.id == "F-0001" && r.relation == "conflicts"), "{:?}", item.related);
    let id = f.gate(&llm).approve(queue_id, None).unwrap();
    assert_eq!(registry::conflicts(&f.conn, MODEL, &id).unwrap(), ["F-0001"]);
    let read = brain::read(&f.conn, &WordEmbedder, &f.checked(), "attention", 4000).unwrap();
    assert!(read.text.contains("⚠ conflicts with F1"), "{}", read.text);

    // 4. Rejected claims aren't proposed again.
    let other = "Linear attention variants reduce the cost of attention.";
    let Admission::Queued { queue_id } = propose(&f, &llm, other, lin.clone(), None) else { panic!() };
    f.gate(&llm).reject(queue_id).unwrap();
    assert!(matches!(propose(&f, &llm, other, lin.clone(), None), Admission::Dropped { rule, .. } if rule == "rejected_before"));

    // Deleting a fact tombstones it and blocks it from coming back.
    f.gate(&llm).delete_fact(&id, None).unwrap();
    assert_eq!(registry::get(&f.conn, MODEL, &id).unwrap().unwrap().status, "deleted");
    assert!(!f.file_text().contains(&id));
    assert_eq!(registry::next_id(&f.conn, MODEL).unwrap(), "F-0003", "IDs are never reused");
}

#[test]
fn integrity_checks_catch_hand_edits_unknown_missing_deleted_and_stale_facts() {
    let mut f = Fixture::new();
    let llm = Scripted::new(vec![], "");
    let quad = cite(&f, "quadratic", "takes quadratic time in the sequence length");
    let Admission::Queued { queue_id } = propose(&f, &llm, CLAIM, quad.clone(), None) else { panic!() };
    f.gate(&llm).approve(queue_id, None).unwrap();
    let path = file::path_for(&f.ws, MODEL);
    let original = f.file_text();

    // A hand edit whose quote still verifies is accepted (registry updated).
    std::fs::write(&path, original.replace("Self-attention takes quadratic time", "Self-attention needs quadratic time")).unwrap();
    assert!(f.checked().ok(), "{:?}", f.checked().problems);
    assert_eq!(registry::get(&f.conn, MODEL, "F-0001").unwrap().unwrap().claim, "Self-attention needs quadratic time in the sequence length.");
    // One whose quote doesn't verify is excluded.
    let edited = f.file_text().replace("takes quadratic time in the sequence length\"", "takes no time at all\"");
    std::fs::write(&path, &edited).unwrap();
    let c = f.checked();
    assert!(c.problems.iter().any(|p| p.kind == "edited"), "{:?}", c.problems);
    assert!(c.usable().is_empty());

    // Unknown ID, a dangling thought, an invalid line, a duplicate.
    let good = registry::get(&f.conn, MODEL, "F-0001").unwrap().unwrap().line;
    let unknown = good.replace("F-0001", "F-0042");
    std::fs::write(&path, format!("# Brain · {MODEL}\n\n## Unsorted\n{good}\n{unknown}\n[THOUGHT] See F-0077.\nstray text\n")).unwrap();
    let kinds: Vec<String> = f.checked().problems.iter().map(|p| p.kind.clone()).collect();
    assert!(kinds.contains(&"unknown_id".into()) && kinds.contains(&"dangling_mention".into()) && kinds.contains(&"invalid_line".into()), "{kinds:?}");
    std::fs::write(&path, format!("# Brain · {MODEL}\n{good}\n{good}\n")).unwrap();
    assert!(f.checked().problems.iter().any(|p| p.kind == "duplicate_id"));

    // Missing from the file → reported; Restore puts the registered line back byte for byte.
    std::fs::write(&path, format!("# Brain · {MODEL}\n\n## Unsorted\n")).unwrap();
    assert!(f.checked().problems.iter().any(|p| p.kind == "missing"));
    f.gate(&llm).restore_fact("F-0001").unwrap();
    assert!(f.file_text().contains(&good));
    assert!(f.checked().ok());

    // The cited document leaves corpus/: the fact goes stale; it comes back with the file.
    let attention = f.ws.corpus_dir().join("attention.txt");
    let bytes = std::fs::read(&attention).unwrap();
    std::fs::remove_file(&attention).unwrap();
    f.sync();
    let c = f.checked();
    assert!(c.problems.iter().any(|p| p.kind == "stale"), "{:?}", c.problems);
    assert!(brain::read(&f.conn, &WordEmbedder, &c, "attention", 4000).unwrap().included.is_empty());
    std::fs::write(&attention, bytes).unwrap();
    f.sync();
    assert!(f.checked().ok());
}

#[test]
fn organize_rearranges_without_touching_fact_lines_and_can_be_undone() {
    let f = Fixture::new();
    let mut llm = Scripted::new(vec![], "");
    for (needle, quote, claim) in [("quadratic", "takes quadratic time in the sequence length", CLAIM), ("linear time", "reduce the cost of attention to linear time", "Linear attention variants reduce the cost of attention to linear time.")] {
        let Admission::Queued { queue_id } = propose(&f, &llm, claim, cite(&f, needle, quote), None) else { panic!() };
        f.gate(&llm).approve(queue_id, None).unwrap();
    }
    let before = f.file_text();
    llm.organize = json!({ "lines": [
        format!("# Brain · {MODEL}"), "", "## Cost of attention",
        "[THOUGHT] F-0001 and F-0002 describe different mechanisms.",
        "[FACT F-0002]", "[FACT F-0001]"
    ] });
    let checked = f.checked();
    let r = organize::organize(&f.ws, &llm, MODEL, 16_384, &checked).unwrap();
    assert!(r.changed && r.errors.is_empty(), "{r:?}");
    let after = f.file_text();
    assert!(after.contains("## Cost of attention\n[THOUGHT] F-0001 and F-0002 describe different mechanisms.\n[FACT] F-0002"), "{after}");
    let facts = |t: &str| { let mut v: Vec<String> = t.lines().filter(|l| l.starts_with("[FACT]")).map(str::to_string).collect(); v.sort(); v };
    assert_eq!(facts(&before), facts(&after), "fact lines are byte-for-byte the same");
    assert!(f.checked().ok());
    // The model never saw hashes, and saw thoughts and facts by ID.
    let reqs = llm.requests.borrow();
    let shown = &reqs.last().unwrap().messages[1].content;
    assert!(shown.contains("[FACT F-0001] Self-attention") && !shown.contains("<doc:"), "{shown}");
    drop(reqs);

    file::undo(&f.ws, MODEL).unwrap();
    assert_eq!(f.file_text(), before);
    assert!(file::undo(&f.ws, MODEL).is_err(), "nothing left to undo");

    // Invalid layouts are refused: a dropped fact, an invented ID, rewritten fact text.
    let lines = file::load(&f.ws, MODEL).unwrap().lines;
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let e = organize::apply_layout(&lines, &s(&["## A", "[FACT F-0001]"])).unwrap_err();
    assert!(e.iter().any(|x| x.contains("F-0002] is missing")), "{e:?}");
    let e = organize::apply_layout(&lines, &s(&["[FACT F-0001]", "[FACT F-0002]", "[FACT F-0009]"])).unwrap_err();
    assert!(e.iter().any(|x| x.contains("F-0009")), "{e:?}");
    let e = organize::apply_layout(&lines, &s(&["[FACT F-0001]", "[FACT F-0001]", "[FACT F-0002]"])).unwrap_err();
    assert!(e.iter().any(|x| x.contains("2 times")), "{e:?}");
    let e = organize::apply_layout(&lines, &s(&["[FACT] F-0001 | rewritten | x", "[FACT F-0002]"])).unwrap_err();
    assert!(!e.is_empty());

    // A failing model gets one retry, then nothing is saved.
    llm.organize = json!({ "lines": ["[FACT F-0001]"] });
    let r = organize::organize(&f.ws, &llm, MODEL, 16_384, &f.checked()).unwrap();
    assert!(!r.changed && r.attempts == 2 && !r.errors.is_empty());
    assert_eq!(f.file_text(), before);
    assert!(matches!(file::load(&f.ws, MODEL).unwrap().lines[0], Line::Heading { level: 1, .. }));
}

#[test]
fn a_report_organizes_first_reads_the_whole_brain_and_proposes_nothing() {
    let f = Fixture::new();
    let mut llm = Scripted::new(vec![], "");
    for (needle, quote, claim) in [("quadratic", "takes quadratic time in the sequence length", CLAIM), ("linear time", "reduce the cost of attention to linear time", "Linear attention variants reduce the cost of attention to linear time.")] {
        let Admission::Queued { queue_id } = propose(&f, &llm, claim, cite(&f, needle, quote), None) else { panic!() };
        f.gate(&llm).approve(queue_id, None).unwrap();
    }
    llm.organize = json!({ "lines": [format!("# Brain · {MODEL}"), "", "## Cost of attention", "[THOUGHT] F-0001 and F-0002 describe different mechanisms.", "[FACT F-0001]", "[FACT F-0002]"] });
    // The two program-made calls (organize_brain, read_brain_file) come first; then the model's.
    llm.moves = vec![
        ("pad", json!({})),
        ("pad", json!({})),
        ("propose_fact", json!({ "claim": "x", "citations": ["C1"], "reason": "y" })),
        ("ask_corpus", json!({ "question": "How does self-attention time scale with sequence length?" })),
    ];
    llm.final_text = "# Attention cost\n\nSelf-attention is quadratic [C1], as noted before [F-0001]; linear variants differ [F2].".into();
    let queued_before = brain::queue::pending(&f.conn, MODEL).unwrap().len();
    let r = f.agent(&llm).report("Why is attention expensive, and what helps?", &mut |_| {}).unwrap();

    let tools: Vec<&str> = r.steps.iter().map(|s| s.tool.as_str()).collect();
    assert_eq!(tools, ["organize_brain", "read_brain_file", "propose_fact", "ask_corpus"]);
    assert_eq!(r.steps[0].summary, "reorganized");
    assert!(r.proposals.is_empty() && brain::queue::pending(&f.conn, MODEL).unwrap().len() == queued_before, "a report proposes nothing");
    assert!(r.invalid_markers.is_empty(), "{:?}", r.invalid_markers);
    let vias: Vec<&str> = r.sources.iter().map(|s| s.via.as_str()).collect();
    assert_eq!(vias, ["C1", "F1", "F2"], "facts resolve to their original citations");

    let reqs = llm.requests.borrow();
    let organize_at = reqs.iter().position(|q| q.messages[0].content.contains("You organize your own research notes")).unwrap();
    let report_at = reqs.iter().position(|q| q.messages[0].content.contains("writing a short report")).unwrap();
    assert!(organize_at < report_at, "organize_brain runs before anything else");
    let first = &reqs[report_at];
    let names: Vec<&str> = first.tools.as_ref().unwrap().as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert_eq!(names, ["ask_corpus"], "no propose_fact or read_brain in a report");
    let brain_text = &first.messages.iter().find(|m| m.tool_name.as_deref() == Some("read_brain_file")).unwrap().content;
    assert!(brain_text.contains("## Cost of attention") && brain_text.contains("[THOUGHT] F-0001 and F-0002"), "headings and thoughts are visible: {brain_text}");
    assert!(brain_text.contains("[FACT] F-0001 | Self-attention takes quadratic time") && !brain_text.contains("<doc:"), "{brain_text}");
    let refused = reqs.iter().flat_map(|q| q.messages.iter()).find(|m| m.tool_name.as_deref() == Some("propose_fact")).unwrap();
    assert!(refused.content.contains("isn't available"), "{}", refused.content);
    drop(reqs);

    // Reports are listed apart from research runs, newest first, with their stored result.
    let listed = localrag_core::agent::runs(&f.conn, localrag_core::agent::Mode::Report, 10).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!((listed[0].status.as_str(), listed[0].question.as_str()), ("done", "Why is attention expensive, and what helps?"));
    assert!(localrag_core::agent::runs(&f.conn, localrag_core::agent::Mode::Research, 10).unwrap().is_empty());
    let (_, result) = localrag_core::agent::run_detail(&f.conn, listed[0].id).unwrap().unwrap();
    assert_eq!(result["mode"], "report");
    assert!(result["text"].as_str().unwrap().starts_with("# Attention cost"));
    assert!(localrag_core::agent::delete_run(&f.conn, listed[0].id).unwrap());
}
