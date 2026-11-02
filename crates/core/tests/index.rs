//! Change detection and statuses for `index::sync` (ARCHITECTURE.md §5, Phase 1 exit criteria).

use std::cell::Cell;
use std::path::Path;
use std::time::{Duration, SystemTime};

use localrag_core::config::{ChunkingConfig, GlobalConfig, WorkspaceConfig};
use localrag_core::db;
use localrag_core::embed::Embedder;
use localrag_core::index::{self, Change, Event, Parsers, Status, SyncReport};
use localrag_core::scan::sha256_str;
use localrag_core::workspace::Workspace;
use rusqlite::Connection;

/// Deterministic vectors derived from the text hash; counts how many texts it embedded.
struct FakeEmbedder {
    model: String,
    dim: usize,
    embedded: Cell<usize>,
}

impl FakeEmbedder {
    fn new(model: &str, dim: usize) -> Self {
        FakeEmbedder { model: model.into(), dim, embedded: Cell::new(0) }
    }
}

impl Embedder for FakeEmbedder {
    fn model(&self) -> &str {
        &self.model
    }
    fn embed_documents(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.embedded.set(self.embedded.get() + texts.len());
        Ok(texts
            .iter()
            .map(|t| sha256_str(t).bytes().take(self.dim).map(|b| b as f32 / 255.0).collect())
            .collect())
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
        let root = dir.path().canonicalize().unwrap();
        let config = WorkspaceConfig {
            chunking: ChunkingConfig { target_tokens: 40, overlap: 0.15 },
            max_file_mb: 1,
            ..WorkspaceConfig::default()
        };
        let ws = Workspace { root, config, global: GlobalConfig::default() };
        let conn = db::open(&ws.db_path()).unwrap();
        Fixture { _dir: dir, ws, conn }
    }

    fn corpus(&self) -> std::path::PathBuf {
        self.ws.corpus_dir()
    }

    fn write(&self, rel: &str, content: impl AsRef<[u8]>) {
        let path = self.corpus().join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn sync(&mut self, embedder: &FakeEmbedder) -> (SyncReport, Vec<Event>) {
        let mut events = Vec::new();
        let parsers = Parsers::new(&self.ws);
        let report = index::sync(&self.ws, &mut self.conn, embedder, &parsers, &mut |e| events.push(e.clone())).unwrap();
        (report, events)
    }

    fn status_of(&self, path: &str) -> Status {
        let s: String = self.conn.query_row("SELECT status FROM documents WHERE path = ?1", [path], |r| r.get(0)).unwrap();
        Status::parse(&s)
    }

    fn count(&self, sql: &str) -> i64 {
        self.conn.query_row(sql, [], |r| r.get(0)).unwrap()
    }
}

fn paragraphs(n: usize, tag: &str) -> String {
    (0..n).map(|i| format!("Paragraph {i} of {tag} discusses attention and its cost in detail.\n\n")).collect::<Vec<_>>().concat()
}

fn set_mtime(path: &Path, secs_ahead: u64) {
    let t = SystemTime::now() + Duration::from_secs(secs_ahead);
    filetime::set_file_mtime(path, filetime::FileTime::from_system_time(t)).unwrap();
}

#[test]
fn second_run_is_a_no_op() {
    let mut f = Fixture::new();
    f.write("a.md", format!("# Notes\n{}", paragraphs(6, "a")));
    f.write("sub/b.txt", paragraphs(3, "b"));
    let e = FakeEmbedder::new("m", 8);

    let (r, _) = f.sync(&e);
    assert_eq!((r.new, r.failed), (2, 0));
    let embedded = e.embedded.get();
    assert!(embedded > 2);
    assert_eq!(f.count("SELECT count(*) FROM chunks"), f.count("SELECT count(*) FROM chunks_vec"));

    let (r, events) = f.sync(&e);
    assert!(!r.changed_any(), "{r:?}");
    assert_eq!(r.unchanged, 2);
    assert!(events.is_empty());
    assert_eq!(e.embedded.get(), embedded, "nothing re-embedded");
}

#[test]
fn touch_without_content_change_only_updates_mtime() {
    let mut f = Fixture::new();
    f.write("a.txt", paragraphs(3, "a"));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    let before = e.embedded.get();

    set_mtime(&f.corpus().join("a.txt"), 60);
    let (r, events) = f.sync(&e);
    assert_eq!(r.touched, 1);
    assert_eq!(events[0].change, Change::Touched);
    assert_eq!(e.embedded.get(), before);

    let (r, _) = f.sync(&e);
    assert_eq!(r.unchanged, 1, "new mtime was recorded");
}

#[test]
fn editing_one_file_reprocesses_only_that_file_and_reuses_vectors() {
    let mut f = Fixture::new();
    f.write("a.txt", paragraphs(8, "a"));
    f.write("b.txt", paragraphs(8, "b"));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    let before = e.embedded.get();

    f.write("a.txt", format!("{}Paragraph appended at the end of a, entirely new text.\n", paragraphs(8, "a")));
    set_mtime(&f.corpus().join("a.txt"), 60);
    let (r, events) = f.sync(&e);
    assert_eq!((r.changed, r.unchanged), (1, 1));
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].path, "a.txt");
    assert!(r.chunks_reused > 0, "unchanged chunks keep their vectors: {r:?}");
    assert_eq!(e.embedded.get() - before, r.chunks_embedded);
    assert_eq!(f.count("SELECT count(*) FROM chunks"), f.count("SELECT count(*) FROM chunks_vec"));
}

#[test]
fn removed_documents_keep_chunks_and_restore_on_rename() {
    let mut f = Fixture::new();
    f.write("paper.txt", paragraphs(4, "paper"));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    let chunks = f.count("SELECT count(*) FROM chunks");

    std::fs::rename(f.corpus().join("paper.txt"), f.ws.root.join("paper.txt")).unwrap();
    let (r, _) = f.sync(&e);
    assert_eq!(r.removed, 1);
    assert_eq!(f.status_of("paper.txt"), Status::Removed);
    assert_eq!(f.count("SELECT count(*) FROM chunks"), chunks, "chunks kept for citations");

    let before = e.embedded.get();
    std::fs::create_dir_all(f.corpus().join("moved")).unwrap();
    std::fs::rename(f.ws.root.join("paper.txt"), f.corpus().join("moved/renamed.txt")).unwrap();
    let (r, events) = f.sync(&e);
    assert_eq!((r.restored, r.new), (1, 0));
    assert_eq!(events[0].change, Change::Restored { from: "paper.txt".into() });
    assert_eq!(f.status_of("moved/renamed.txt"), Status::Indexed);
    assert_eq!(e.embedded.get(), before, "restore does not re-embed");
}

#[test]
fn rename_within_one_run_is_detected() {
    let mut f = Fixture::new();
    f.write("old.txt", paragraphs(2, "x"));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    std::fs::rename(f.corpus().join("old.txt"), f.corpus().join("new.txt")).unwrap();
    let (r, _) = f.sync(&e);
    assert_eq!((r.restored, r.new, r.removed), (1, 0, 0));
    assert_eq!(f.count("SELECT count(*) FROM documents"), 1);
}

#[test]
fn forget_only_removed_and_leaves_no_orphans() {
    let mut f = Fixture::new();
    f.write("keep.txt", paragraphs(2, "keep"));
    f.write("drop.txt", paragraphs(4, "drop"));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    let keep_id = index::find_document(&f.conn, "keep.txt").unwrap().unwrap();
    let drop_id = index::find_document(&f.conn, "corpus/drop.txt").unwrap().unwrap();

    let err = index::forget(&f.ws, &f.conn, keep_id).unwrap_err().to_string();
    assert!(err.contains("only removed documents"), "{err}");

    std::fs::remove_file(f.corpus().join("drop.txt")).unwrap();
    f.sync(&e);
    index::forget(&f.ws, &f.conn, drop_id).unwrap();
    assert_eq!(f.count("SELECT count(*) FROM documents"), 1);
    let chunks = f.count("SELECT count(*) FROM chunks");
    assert_eq!(f.count("SELECT count(*) FROM chunks_vec"), chunks);
    assert_eq!(f.count("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'drop'"), 0);
    assert!(f.count("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'keep'") > 0);
}

#[test]
fn switching_embedding_model_reembeds_everything() {
    let mut f = Fixture::new();
    f.write("a.txt", paragraphs(5, "a"));
    let e1 = FakeEmbedder::new("model-one", 8);
    f.sync(&e1);
    let chunks = f.count("SELECT count(*) FROM chunks") as usize;

    let e2 = FakeEmbedder::new("model-two", 4);
    let (r, _) = f.sync(&e2);
    assert_eq!(r.reembedded, chunks);
    assert_eq!(db::get_meta(&f.conn, "embed_model").unwrap().as_deref(), Some("model-two"));
    assert_eq!(db::get_meta(&f.conn, "embed_dim").unwrap().as_deref(), Some("4"));
    assert_eq!(f.count("SELECT count(*) FROM chunks_vec"), chunks as i64);
    let (r, _) = f.sync(&e2);
    assert_eq!(r.reembedded, 0);
}

#[test]
fn ignore_rules_hidden_files_and_size_cap() {
    let mut f = Fixture::new();
    f.write("ok.md", "# Title\nBody text.\n");
    f.write(".hidden.txt", "secret");
    f.write("drafts/skip.txt", "draft");
    f.write("image.png", "not a document");
    f.write("huge.txt", "x".repeat(2 * 1024 * 1024));
    f.write(".localragignore", "drafts/\n");
    let e = FakeEmbedder::new("m", 8);
    let (r, _) = f.sync(&e);
    assert_eq!(r.new, 1);
    assert_eq!(r.too_large.iter().map(|t| t.0.as_str()).collect::<Vec<_>>(), ["huge.txt"]);
    assert_eq!(f.count("SELECT count(*) FROM documents"), 1);
}

#[test]
fn bad_files_fail_once_and_retry_only_when_changed() {
    let mut f = Fixture::new();
    f.write("bad.txt", [0xff, 0xfe, 0x00, 0x80]);
    let e = FakeEmbedder::new("m", 8);
    let (r, events) = f.sync(&e);
    assert_eq!(r.failed, 1);
    assert!(matches!(&events[0].change, Change::Failed(m) if m.contains("UTF-8")));
    assert_eq!(f.status_of("bad.txt"), Status::Failed);

    let (r, _) = f.sync(&e);
    assert_eq!((r.failed, r.unchanged), (0, 1), "not retried while unchanged");

    f.write("bad.txt", "Now valid text.\n");
    set_mtime(&f.corpus().join("bad.txt"), 60);
    let (r, _) = f.sync(&e);
    assert_eq!(r.changed, 1);
    assert_eq!(f.status_of("bad.txt"), Status::Indexed);
}

/// Needs libpdfium (scripts/fetch-pdfium.sh); skipped when it isn't available.
#[test]
fn pdfs_get_page_labels_and_scanned_pages_are_flagged() {
    use_built_helper();
    let Ok(lib) = localrag_core::parse::pdf::find_library(None) else {
        eprintln!("skipping: libpdfium not found");
        return;
    };
    let pdfium = localrag_core::parse::pdf::pdfium(&lib).unwrap();
    let mut f = Fixture::new();
    f.write("sample.pdf", localrag_core::doctor::SAMPLE_PDF);
    f.write("scanned.pdf", scanned_pdf(pdfium));
    let e = FakeEmbedder::new("m", 8);
    let (r, _) = f.sync(&e);
    assert_eq!((r.new, r.failed, r.needs_ocr), (2, 0, 1), "{r:?}");

    assert_eq!(f.status_of("sample.pdf"), Status::Indexed);
    let (label, pdf_page): (String, i64) = f.conn.query_row(
        "SELECT c.page_label_start, c.pdf_page_start FROM chunks c JOIN documents d ON d.id = c.doc_id
         WHERE d.path = 'sample.pdf' AND c.text LIKE '%quadratic%'",
        [],
        |r| Ok((r.get(0)?, r.get(1)?)),
    ).unwrap();
    assert_eq!((label.as_str(), pdf_page), ("345", 1));

    assert_eq!(f.status_of("scanned.pdf"), Status::NeedsOcr);
    let scanned: String = f.conn.query_row("SELECT scanned_pages FROM documents WHERE path = 'scanned.pdf'", [], |r| r.get(0)).unwrap();
    assert_eq!(scanned, "[1]");
}

/// One page containing only an image of rendered text, like a scan.
fn scanned_pdf(pdfium: &pdfium_render::prelude::Pdfium) -> Vec<u8> {
    use pdfium_render::prelude::*;
    let source = pdfium.load_pdf_from_byte_slice(localrag_core::doctor::SAMPLE_PDF, None).unwrap();
    let image = source.pages().get(0).unwrap()
        .render_with_config(&PdfRenderConfig::new().set_target_width(612)).unwrap()
        .as_image().unwrap();
    let mut doc = pdfium.create_new_pdf().unwrap();
    let mut page = doc.pages_mut().create_page_at_end(PdfPagePaperSize::from_points(PdfPoints::new(612.0), PdfPoints::new(792.0))).unwrap();
    page.objects_mut()
        .create_image_object(PdfPoints::ZERO, PdfPoints::ZERO, &image, Some(PdfPoints::new(612.0)), Some(PdfPoints::new(792.0)))
        .unwrap();
    doc.save_to_bytes().unwrap()
}

#[test]
fn damaged_pdfs_fail_cleanly_through_the_helper() {
    use_built_helper();
    if localrag_core::parse::pdf::find_library(None).is_err() {
        return;
    }
    let mut f = Fixture::new();
    f.write("broken.pdf", b"%PDF-1.4\n1 0 obj << garbage that is not a PDF structure");
    f.write("fake.pdf", b"just text with a .pdf name");
    let e = FakeEmbedder::new("m", 8);
    let (r, events) = f.sync(&e);
    assert_eq!(r.failed, 2, "{events:?}");
    let msg = |p: &str| events.iter().find(|e| e.path == p).map(|e| format!("{:?}", e.change)).unwrap();
    assert!(msg("broken.pdf").contains("damaged or not a valid PDF"), "{}", msg("broken.pdf"));
    assert!(msg("fake.pdf").contains("no PDF header"), "{}", msg("fake.pdf"));
}

#[test]
fn concurrent_first_open_migrates_once() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("index.db");
    let handles: Vec<_> = (0..8).map(|_| {
        let p = path.clone();
        std::thread::spawn(move || db::open(&p).map(|_| ()))
    }).collect();
    for h in handles {
        h.join().unwrap().unwrap();
    }
}

/// PDFs are parsed by the `localrag-pdf` helper binary built alongside these tests.
fn use_built_helper() {
    // SAFETY: tests that read it all set the same value.
    unsafe { std::env::set_var("LOCALRAG_PDF_HELPER", env!("CARGO_BIN_EXE_localrag-pdf")) };
}

#[test]
fn rename_updates_references_without_reindexing() {
    let mut f = Fixture::new();
    f.write("Parry et al. - 2011 - Survivors.txt", paragraphs(3, "survivors"));
    f.write("other.txt", paragraphs(2, "other"));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    let embedded = e.embedded.get();
    let id = index::find_document(&f.conn, "Parry et al. - 2011 - Survivors.txt").unwrap().unwrap();

    let session = localrag_core::history::create_session(&f.conn, "q", "m").unwrap();
    let snaps = serde_json::json!([{ "ordinal": 0, "marker": 1, "label": "Parry et al. 2011, ll. 1–3",
        "doc_path": "Parry et al. - 2011 - Survivors.txt", "doc_sha256": "x", "chunk_text_sha": "y", "chunk_id": 1,
        "quote": null, "quote_verified": false }]);
    f.conn.execute(
        "INSERT INTO messages (session_id, role, content, citations_json, created_at) VALUES (?1, 'assistant', 'A [S1].', ?2, 'now')",
        rusqlite::params![session, snaps.to_string()],
    ).unwrap();
    std::fs::write(f.ws.state_dir().join("eval.jsonl"), "{\"question\": \"q\", \"expected_doc\": \"Parry et al\"}\n{\"question\": \"r\", \"expected_doc\": \"other\"}\n").unwrap();

    let err = index::rename_document(&f.ws, &mut f.conn, id, "parry.pdf").unwrap_err().to_string();
    assert!(err.contains("keep the .txt extension"), "{err}");
    let err = index::rename_document(&f.ws, &mut f.conn, id, "other.txt").unwrap_err().to_string();
    assert!(err.contains("already exists"), "{err}");

    let r = index::rename_document(&f.ws, &mut f.conn, id, "parry-2011.txt").unwrap();
    assert_eq!((r.path.as_str(), r.label.as_str()), ("parry-2011.txt", "parry-2011"));
    assert_eq!((r.citations_updated, r.eval_items_updated), (1, 1));
    assert!(f.corpus().join("parry-2011.txt").is_file() && !f.corpus().join("Parry et al. - 2011 - Survivors.txt").exists());

    let msg: String = f.conn.query_row("SELECT citations_json FROM messages", [], |r| r.get(0)).unwrap();
    let snap: serde_json::Value = serde_json::from_str::<Vec<serde_json::Value>>(&msg).unwrap().remove(0);
    assert_eq!((snap["doc_path"].as_str(), snap["label"].as_str()), (Some("parry-2011.txt"), Some("parry-2011, ll. 1–3")));
    let eval = std::fs::read_to_string(f.ws.state_dir().join("eval.jsonl")).unwrap();
    assert!(eval.contains("\"expected_doc\":\"parry-2011.txt\"") && eval.contains("\"expected_doc\": \"other\""), "{eval}");

    // The next sync sees nothing to do: same content, same stat, new path already recorded.
    let (report, events) = f.sync(&e);
    assert!(!report.changed_any() && events.is_empty(), "{report:?} {events:?}");
    assert_eq!(e.embedded.get(), embedded);
}

/// Scanned page → needs_ocr → Vision OCR (cached) → reindexed with OCR text → highlightable.
#[test]
fn ocr_fills_scanned_pages_and_is_cached() {
    use_built_helper();
    let Ok(lib) = localrag_core::parse::pdf::find_library(None) else { return };
    let pdfium = localrag_core::parse::pdf::pdfium(&lib).unwrap();
    let mut f = Fixture::new();
    f.write("scan.pdf", scanned_pdf(pdfium));
    let e = FakeEmbedder::new("m", 8);
    f.sync(&e);
    assert_eq!(f.status_of("scan.pdf"), Status::NeedsOcr);

    let parsers = Parsers::new(&f.ws);
    let waiting = index::needing_ocr(&f.conn).unwrap();
    let (id, path, sha, pages) = waiting[0].clone();
    assert_eq!(pages, [1]);
    let cancel = std::sync::atomic::AtomicBool::new(false);
    let cache = localrag_core::ocr::ocr_document(&f.ws, parsers.pdf_helper().unwrap(), &f.corpus().join(&path), &sha, &pages, &cancel, &mut |_, _| {}).unwrap();
    assert!(cache.pages[&1].text.to_lowercase().contains("quadratic time"), "{}", cache.pages[&1].text);
    assert!(f.ws.ocr_cache_path(&sha).is_file());

    index::mark_for_reprocess(&f.conn, id).unwrap();
    let (r, _) = f.sync(&e);
    assert_eq!(r.changed, 1);
    assert_eq!(f.status_of("scan.pdf"), Status::Indexed);
    let (chunk_id, text, ocr): (i64, String, bool) = f.conn
        .query_row("SELECT id, text, ocr FROM chunks WHERE text LIKE '%quadratic%'", [], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
        .unwrap();
    assert!(ocr, "chunk is flagged as OCR");
    let start = text[..text.find("quadratic").unwrap()].chars().count();
    let h = localrag_core::highlight::for_chunk(&f.ws, &f.conn, chunk_id, Some((start, start + "quadratic time".len()))).unwrap();
    let localrag_core::highlight::Highlight::Pdf { pages } = h else { panic!("{h:?}") };
    assert_eq!(pages[0].pdf_page, 1);
    assert!(!pages[0].rects.is_empty(), "OCR line boxes highlight the quote");

    // Rechunking reuses the cache: no OCR needed again, and the doc stays indexed.
    index::mark_for_reprocess(&f.conn, id).unwrap();
    f.sync(&e);
    assert_eq!(f.status_of("scan.pdf"), Status::Indexed);
}

#[test]
fn metadata_labels_get_suffixes_and_page_ranges_become_printed_pages() {
    use localrag_core::metadata::{self, crossref::{Author, Work}};
    let mut f = Fixture::new();
    f.write("a.txt", paragraphs(2, "a"));
    f.write("b.txt", paragraphs(2, "b"));
    f.sync(&FakeEmbedder::new("m", 8));
    let id = |p: &str| index::find_document(&f.conn, p).unwrap().unwrap();
    let work = |title: &str, page: Option<&str>| Work {
        doi: format!("10.1/{title}"), title: Some(title.into()), year: Some(2021), container: None, kind: None,
        page: page.map(str::to_string), cited_by: None, abstract_text: None, author: vec![Author { family: Some("Smith".into()), given: Some("A".into()), literal: None }],
    };
    metadata::apply_match(&f.conn, id("a.txt"), &work("Alpha study", None), "crossref_doi").unwrap();
    metadata::apply_match(&f.conn, id("b.txt"), &work("Beta study", None), "crossref_doi").unwrap();
    metadata::assign_labels(&f.conn).unwrap();
    let label = |p: &str| -> String { f.conn.query_row("SELECT cite_label FROM documents WHERE path = ?1", [p], |r| r.get(0)).unwrap() };
    assert_eq!((label("a.txt"), label("b.txt")), ("Smith 2021a".into(), "Smith 2021b".into()));

    // A 2-page "PDF" printed as pp. 345–346: printed numbers come from the page range.
    f.conn.execute("UPDATE documents SET kind = 'pdf', page_count = 2 WHERE path = 'a.txt'", []).unwrap();
    f.conn.execute("UPDATE chunks SET pdf_page_start = 1, pdf_page_end = 2 WHERE doc_id = ?1", [id("a.txt")]).unwrap();
    metadata::apply_match(&f.conn, id("a.txt"), &work("Alpha study", Some("345-346")), "crossref_doi").unwrap();
    let (ls, le): (String, String) = f.conn.query_row("SELECT page_label_start, page_label_end FROM chunks WHERE doc_id = ?1 LIMIT 1", [id("a.txt")], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert_eq!((ls.as_str(), le.as_str()), ("345", "346"));

    // Manual edits win and are never overwritten by later lookups.
    metadata::set_manual(&f.conn, id("a.txt"), &metadata::ManualMetadata {
        title: Some("Alpha".into()), authors: vec!["Jones, B".into()], year: Some(1999), doi: None, container: None, page: None,
    }).unwrap();
    metadata::apply_match(&f.conn, id("a.txt"), &work("Overwritten?", None), "crossref_query").unwrap();
    assert_eq!(label("a.txt"), "Jones 1999");
    let src: String = f.conn.query_row("SELECT metadata_source FROM documents WHERE path = 'a.txt'", [], |r| r.get(0)).unwrap();
    assert_eq!(src, "manual");

    // Reprocessing a file (e.g. after an edit) keeps its Crossref metadata.
    std::fs::write(f.corpus().join("b.txt"), paragraphs(3, "b edited")).unwrap();
    set_mtime(&f.corpus().join("b.txt"), 60);
    f.sync(&FakeEmbedder::new("m", 8));
    let (src, lbl): (String, String) = f.conn.query_row("SELECT metadata_source, cite_label FROM documents WHERE path = 'b.txt'", [], |r| Ok((r.get(0)?, r.get(1)?))).unwrap();
    assert_eq!((src.as_str(), lbl.as_str()), ("crossref_doi", "Smith 2021"), "Crossref data survives reprocessing");
}

#[test]
fn bibliographies_are_split_off_counted_and_linked_to_the_corpus() {
    let mut f = Fixture::new();
    let body = paragraphs(6, "synchronous cancers in registries");
    f.write("baicry.md", format!(
        "---\ntitle: What is the most appropriate period to define synchronous cancers\nauthor: Baicry, F.\nyear: 2021\n---\n# Study\n{body}\n# References\n\
         1. Parry C, Kent EE, Mariotto AB. Cancer survivors: a booming population of zebrafinches. Cancer Epidemiol Biomarkers Prev. 2011;20(10):1996–2005.\n\
         2. Utada M, Ohno Y. Incidence of multiple primary cancers and interval between first and second primary cancers. Cancer Sci. 2014;105(7):890–6.\n\
         3. Gass B, Marrer E. Use of a case-mix approach to study the trends in the incidence of second primary cancers. Ann Epidemiol. 2018;28(5):322–7.\n"));
    f.write("parry.md", format!("---\ntitle: Cancer survivors: a booming population of zebrafinches\nauthor: Parry, C.\nyear: 2011\n---\n# Survivors\n{}", paragraphs(3, "survivors")));
    f.write("plain.txt", paragraphs(2, "no bibliography here"));
    f.sync(&FakeEmbedder::new("m", 8));

    let id = |p: &str| index::find_document(&f.conn, p).unwrap().unwrap();
    let count = |p: &str| -> Option<i64> { f.conn.query_row("SELECT reference_count FROM documents WHERE path = ?1", [p], |r| r.get(0)).unwrap() };
    assert_eq!(count("baicry.md"), Some(3));
    assert_eq!(count("plain.txt"), None, "no bibliography → N/A");
    assert_eq!(f.count("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'zebrafinches'"), 0, "the bibliography entry isn't chunked (and front matter never is)");
    assert_eq!(f.count("SELECT count(*) FROM chunks_fts WHERE chunks_fts MATCH 'utada'"), 0, "bibliography entries aren't chunked");

    let refs = localrag_core::references::for_document(&f.conn, id("baicry.md")).unwrap();
    assert_eq!(refs[0].in_corpus.as_ref().map(|v| v["id"].as_i64().unwrap()), Some(id("parry.md")), "reference 1 is a corpus document");
    assert!(refs[1].in_corpus.is_none());
    assert_eq!(refs[1].label, "Utada 2014");

    let missing = localrag_core::references::missing_for_question(&f.conn, None, "What is the interval between first and second primary cancers?", 10).unwrap();
    assert_eq!(missing[0].label, "Utada 2014", "ranked by overlap with the question");
    assert!(missing.iter().all(|m| !m.raw.contains("zebrafinches")), "works already in corpus/ aren't suggested");
}
