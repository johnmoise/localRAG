//! Mapping chunk spans back to PDF rectangles and text offsets (ARCHITECTURE.md §10).

use localrag_core::config::{GlobalConfig, WorkspaceConfig};
use localrag_core::embed::Embedder;
use localrag_core::highlight::{self, Highlight};
use localrag_core::index::{self, Parsers};
use localrag_core::workspace::Workspace;
use localrag_core::{db, doctor};
use rusqlite::Connection;

struct Zero;
impl Embedder for Zero {
    fn model(&self) -> &str {
        "zero"
    }
    fn embed_documents(&self, texts: &[String]) -> anyhow::Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}

fn setup(files: &[(&str, &[u8])]) -> (tempfile::TempDir, Workspace, Connection) {
    let dir = tempfile::tempdir().unwrap();
    Workspace::init(dir.path()).unwrap();
    let ws = Workspace { root: dir.path().canonicalize().unwrap(), config: WorkspaceConfig::default(), global: GlobalConfig::default() };
    for (name, bytes) in files {
        std::fs::write(ws.corpus_dir().join(name), bytes).unwrap();
    }
    let mut conn = db::open(&ws.db_path()).unwrap();
    index::sync(&ws, &mut conn, &Zero, &Parsers::new(&ws), &mut |_| {}).unwrap();
    (dir, ws, conn)
}

fn chunk_with(conn: &Connection, needle: &str) -> (i64, String) {
    conn.query_row("SELECT id, text FROM chunks WHERE text LIKE ?1", [format!("%{needle}%")], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
}

fn char_span(text: &str, needle: &str) -> (usize, usize) {
    let start = text[..text.find(needle).unwrap()].chars().count();
    (start, start + needle.chars().count())
}

#[test]
fn pdf_quote_and_whole_chunk_map_to_page_rectangles() {
    if localrag_core::parse::pdf::find_library(None).is_err() {
        eprintln!("skipping: libpdfium not found");
        return;
    }
    let (_d, ws, conn) = setup(&[("sample.pdf", doctor::SAMPLE_PDF)]);
    let (id, text) = chunk_with(&conn, "quadratic time");

    let quote = highlight::for_chunk(&ws, &conn, id, Some(char_span(&text, "quadratic time"))).unwrap();
    let Highlight::Pdf { pages } = quote else { panic!("{quote:?}") };
    assert_eq!(pages.len(), 1);
    assert_eq!(pages[0].pdf_page, 1);
    assert_eq!(pages[0].rects.len(), 1, "one line → one merged rectangle: {:?}", pages[0].rects);
    let [x, y, w, h] = pages[0].rects[0];
    // "quadratic time" sits on the second line, part-way across, near the top of the page.
    assert!(x > 0.2 && x < 0.7 && w > 0.05 && w < 0.4, "{:?}", pages[0].rects[0]);
    assert!(y > 0.05 && y < 0.15 && h > 0.01 && h < 0.04, "{:?}", pages[0].rects[0]);

    let whole = highlight::for_chunk(&ws, &conn, id, None).unwrap();
    let Highlight::Pdf { pages } = whole else { panic!() };
    assert!(pages.iter().map(|p| p.rects.len()).sum::<usize>() >= 3, "every line of the chunk: {pages:?}");
}

#[test]
fn text_quote_maps_to_file_offsets() {
    let md = "# Notes\nSelf-attention compares every token with every\nother token, so its cost grows quadratically.\n";
    let (_d, ws, conn) = setup(&[("notes.md", md.as_bytes())]);
    let (id, text) = chunk_with(&conn, "quadratically");
    // The quote crosses a line break in the file; chunk text has a space there.
    let h = highlight::for_chunk(&ws, &conn, id, Some(char_span(&text, "every other token"))).unwrap();
    let Highlight::Text { start, end } = h else { panic!("{h:?}") };
    let file: Vec<char> = md.chars().collect();
    assert_eq!(file[start..end].iter().collect::<String>(), "every\nother token");
}

#[test]
fn changed_or_removed_files_are_reported_unavailable() {
    let (_d, ws, conn) = setup(&[("a.txt", b"Some text about attention costs.\n")]);
    let (id, _) = chunk_with(&conn, "attention");
    std::fs::write(ws.corpus_dir().join("a.txt"), "Edited since indexing.\n").unwrap();
    assert!(matches!(highlight::for_chunk(&ws, &conn, id, None).unwrap(), Highlight::Unavailable { .. }));
}
