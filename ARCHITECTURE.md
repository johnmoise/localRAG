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
| Chat history | Kept per workspace | 8 |
| Context size | `num_ctx` chosen from RAM (16k on 16 GB), fixed per model, sent on every request | 11 |
| Over budget | Submit is blocked with "Over token count by X tokens". No live counter | 11 |
| Context overload | Always warn when anything is dropped from context | 11 |
| Reasoning agent | Tools `read_brain`, `ask_corpus`, `propose_fact`, `organize_brain` | 12 |
| Brains | One Markdown file per reasoning model. `[FACT]` lines are immutable to the model; headings and `[THOUGHT]` lines are visible to it only in `organize_brain` | 12 |
| Fact review | Proposed facts wait in a queue for your approval. Rejected or deleted facts are never proposed again | 12 |
| Reference integrity | Every fact ID and chunk reference is resolved and checked; unresolvable ones are excluded and reported | 13 |
| UI | Four tabs (Ask, Search, Library, Second Brain) plus Settings | 15 |

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
| Files | `ignore` (gitignore-style rules), `notify` (watching) |
| Config | `toml` + `serde` |
| Logging / errors | `tracing`, `anyhow` / `thiserror` |
| Frontend | Static HTML/JS/CSS embedded with `rust-embed`; vendored `marked`, `DOMPurify`, PDF.js. No build step, works offline |
| Desktop shell | Tauri 2: a native window around the same UI and an in-process server |

**Licenses (binaries will be redistributed):** PDFium (BSD/Apache), PDF.js (Apache-2.0),
sqlite-vec (MIT/Apache) and Tauri (MIT/Apache) are fine. Avoid MuPDF (AGPL).

## 4. Workspace & on-disk layout

The directory `localrag` runs in is the workspace. Like git, it walks up from the CWD to find
`.localrag/`. `localrag init` creates `.localrag/` and `corpus/`, and offers to add
`.localrag/` to `.gitignore`.

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
- **Forget** (Library, or `localrag forget <doc>`) permanently deletes the record, chunks,
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
   - With `ask`, the Library shows **Run OCR**, since a 300-page book takes minutes.
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
   - Otherwise it's shown in the Library as a *proposed* match to confirm or reject.
3. **No match or offline** → PDF metadata → first-page heuristics → manual edit.

- **Precedence:** manual > Crossref DOI > confirmed Crossref query > PDF metadata >
  heuristics. `metadata_source` is shown in the Library.
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
- **Cache:** responses are cached in `~/.cache/localrag/crossref/`, keyed by DOI or normalized
  query. They're refreshed only by **Refresh metadata**.

**Email**
- **First load:** a dialog explains the polite pool and asks for an email, with **Skip (use
  public pool)**. After a skip it doesn't ask again; Settings shows a note instead.
- **Settings:** view, change or clear the email, with format validation. Also available as
  `localrag config set crossref_email …`.
- Stored in the global config and sent only to Crossref.

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
  citations_json TEXT,     -- snapshot: label, page, quote, doc sha256, chunk text_sha
  warnings_json TEXT,      -- context-overload warnings (§11)
  supported INTEGER, model TEXT, created_at TEXT
);

-- Brains (§12, §13) -----------------------------------------------------------------------
CREATE TABLE brain_facts (                -- registry: the source of truth for fact IDs
  model TEXT, fact_id TEXT,               -- e.g. 'deepseek-r1:8b', 'F-0012'
  line_sha TEXT,                          -- hash of the approved [FACT] line
  status TEXT,                            -- active|stale|broken|deleted
  added_at TEXT, session_id INTEGER, reason TEXT,
  PRIMARY KEY (model, fact_id)
);
CREATE TABLE fact_citations (
  model TEXT, fact_id TEXT, ord INTEGER,
  doc_sha256 TEXT, chunk_text_sha TEXT, quote TEXT,
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
  status TEXT,                            -- pending|approved|rejected
  created_at TEXT, decided_at TEXT
);
CREATE TABLE rejected_facts (             -- never propose these again
  id INTEGER PRIMARY KEY, model TEXT, claim TEXT, citations_json TEXT,
  origin TEXT,                            -- queue_reject|deleted
  at TEXT
);
CREATE VIRTUAL TABLE brain_vec USING vec0(embedding float[1024]);   -- active + rejected fact embeddings
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
- **No usable quote:** the chunk sentence most similar to the claim is highlighted instead,
  in a distinct color.
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

**Security.** Binding to `127.0.0.1` isn't enough. Any website in your browser can send
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
| `GET /api/documents[/{id}]` | Library listing and details |
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
| `PATCH/DELETE /api/brain/{model}/facts/{id}` | Edit (re-verified) / delete (→ rejected) a fact |
| `POST /api/brain/{model}/organize` · `/undo` · `/check` | Organize, undo the last organize, re-check integrity |
| `GET /api/brain/{model}/queue` | Proposed facts |
| `POST /api/brain/{model}/queue/{id}/{approve\|reject\|merge}` | Queue actions (`approve` accepts an edited claim) |
| `POST /api/agent/run` | (Phase 7) Run the reasoning agent; SSE adds `step`, `tool_call`, `tool_result`, `brain_update` |

## 15. User interface

One page with four tabs, a Settings panel, and a first-load dialog.

- **First load:** the Crossref email dialog, if no email is set (§7).
- **Ask:**
  - Chat with a model dropdown and session list.
  - An over-budget Submit shows "Over token count by X tokens".
  - Answers show context-overload banners, flagged sentences, and citation chips
    `[Smith 2021, p. 12]`. A chip opens a side panel with the chunk, the highlighted quote and
    **Open at p. 12** (PDF.js with the highlight).
- **Search:** ranked passages, filterable by document or year.
- **Library:**
  - Documents with status and metadata source.
  - Actions: Run OCR, Forget (removed only, confirmed), Forget all removed, edit metadata,
    confirm/reject a Crossref match, refresh metadata, reindex.
- **Second Brain** (Phase 7):
  - **Reasoning model selector** (installed models with `thinking` + `tools`). Models that have
    a brain file but are no longer installed appear read-only.
  - **Main pane:** the selected model's brain in its current structure.
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
  - **Build note:** `image` is pinned to 0.25.6, because 0.25.9 requires an unreleased `png`.

**Phase 1: workspace & stateful indexing (CLI)**
- `init`, config, `corpus/`, ignore rules; scan/diff; all statuses; restore; `forget`.
- PDF/MD/TXT parsing, chunking, embedding; `index`, `status`.
- *Exit:* a second run reports 0 changed in about a second; touching one file affects only
  that file.

**Phase 2: retrieval & grounded answers (CLI)**
- Hybrid search (`search`); budget check and warnings; prompting; quote verification;
  "I don't know" (`ask`).
- Chat-history citation snapshots; first eval set; `eval`; `check` for chat-history
  references.
- *Exit:* correct pages and verified quotes on the eval; a recorded baseline refusal rate.

**Phase 3: web app**
- axum, token auth, SSE; Ask / Search / Library; model dropdown; over-budget error; warning
  banners; chat history; PDF.js with highlights; `serve`.
- *Exit:* the full loop in the browser across two workspaces, with no state leaking.

**Phase 4: metadata & OCR**
- DOI extraction; Crossref client (limiter, cache, background queue); page labels;
  `[Author Year, p. N]`.
- Email dialog and Settings; proposed-match review; manual edits.
- Vision OCR with cache, Run OCR and progress.

**Phase 5: quality & polish**
- Follow-up rewriting, file watcher, strict mode.
- Tune the gate, k and chunk size against the eval.

**Phase 6: packaging** (§19)
- Tauri app, bundled PDFium, first-run checks, ad-hoc signing, `.dmg` + README.

**Phase 7: reasoning agent & second brain**
- Brain files, registry, integrity checks (§13), `check` for brains.
- Second Brain tab with the Proposed Facts sidebar.
- Tools and agent loop; admission rules; `organize_brain` with undo.
- *Exit:* an agent run proposes facts; approved facts appear in the right model's brain;
  `organize_brain` restructures without changing any fact line; `check` is clean.

**Later**
- Grounded "add this paper" suggestions from reference lists; bibliography export.
- Vision-model OCR for hard pages; reranker.
- An MCP server exposing `search` / `ask`.

## 19. Packaging & distribution

- **Build:** `aarch64-apple-darwin` only. Tauri 2 bundles the UI, the in-process server and
  `libpdfium.dylib` (in `Contents/Frameworks`) into a `.app` inside a `.dmg`.
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

---

## Appendix A: Original notes

I want to build a program that helps the user explore and more efficiently utilize
literature about that particular topic to conduct research and overall edify their own
understanding of the material.
- It should run a basic server that displays a simple web frontend to localhost. This
  web front end should allow the user to enter questions about the material/literature
  that is indexed in the RAG, search through the documents, find relevant chunks of
  material and then have a locally running AI model answer the user's question with
  these chunks as its context. As a RAG, the model should never provide unsubstantiated
  answers and should always provide citations to corroborate.
- The frontend should also allow the user to select from currently installed ollama models
  for use with the RAG.
- As a RAG, all answers should be cited from the literature, and if the model is unable
  to substantiate its answer, it should state that it does not know and make suggestions
  for material that it should be provided that it might find helpful in forming a
  correct answer.
- It should be stateful. (i.e. if the program is rerun, it should check for
  differences/changes in files, but provided there is no change, it should not reindex.
- I have a number of different interests and research projects, so I envision that I
  would likely run this program in different workspaces, each with its own corpus of
  documents for the local RAG. The indexing and model contexts should remain distinct to
  the workspace/directory in which the executable is run.
- Out of scope for now, but I hope to implement a reasoning model component to this
  program as well. The user can pose a question to the reasoning model, the reasoning
  model will think and reason about the question and then ask its own questions to the
  RAG model, whose answers will be used by the reasoning model to substantiate its
  reasoning and conclusions.
