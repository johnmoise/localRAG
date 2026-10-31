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
