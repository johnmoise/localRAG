# localRAG

## Vision & requirements (original notes)

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

---

## Implementation plan

*Drafted 2026-10-31. Everything below is a proposal; see §14 for open questions.*

### 1. Requirements distilled

| #  | Requirement | Implication |
|----|-------------|-------------|
| R1 | Local web UI on localhost | Small HTTP server bound to `127.0.0.1`; frontend with no build step |
| R2 | Ask questions, search documents, see relevant chunks | Two separate paths: **search** (retrieval only) and **ask** (retrieval + generation) |
| R3 | Pick from installed Ollama models | Query Ollama at runtime; the chat model is chosen per request and never baked into the index |
| R4 | Every claim cited; otherwise "I don't know" + suggestions | Layered grounding: retrieval gate → prompt contract → citation validation → optional verification (§8) |
| R5 | Stateful, incremental indexing | Per-file fingerprints in a DB; only changed files are reprocessed (§5) |
| R6 | Per-workspace isolation | All state lives in `<workspace>/.localrag/`; only global defaults are shared (§4) |
| R7 | (Future) reasoning agent that queries the RAG | RAG core is a library returning *structured* answers, usable as a tool (§13) |

On R4: no prompt can *guarantee* a 4–9B local model never makes an unsupported claim. The
realistic goal is to make unsupported claims rare, detectable, and visibly flagged, and to
**measure** the rate with an eval set (§10) instead of hoping.

### 2. Environment snapshot (this machine)

- Apple M2, 16 GB unified memory.
- Python 3.14 (Homebrew), `uv`, Node, Bun.
- Ollama 0.32.15. Chat models: `qwen3.5:{0.8b,4b,9b}` (+ MLX variants), `deepseek-r1:8b`,
  `llama3.2`, `ornith:9b`.
- **No embedding model installed yet**: needs one, e.g. `qwen3-embedding:0.6b`,
  `embeddinggemma`, or `nomic-embed-text` (pick in Phase 0).
- This folder has no git repo of its own; the enclosing repo is rooted at `~`. Run
  `git init` here first.

Memory budget: a 9B chat model (~6.6 GB) + a small embedding model (~0.5–1 GB) + KV cache
for a 16k context fits in 16 GB with little headroom. Default to `qwen3.5:4b` for speed and
offer 9B for quality.

### 3. Tech stack

| Concern | Choice | Notes / alternatives |
|---------|--------|----------------------|
| Language | Python ≥ 3.12 | Best ecosystem for document parsing and Ollama. Pin the interpreter with uv (`.python-version`) to one every dependency ships wheels for; check 3.14 in Phase 0, else use 3.13. |
| Packaging | `uv` project; `uv tool install -e .` | Puts a `localrag` command on PATH, runnable from any directory (R6) |
| CLI | Typer | |
| Server | FastAPI + uvicorn | Async; Server-Sent Events (SSE) for token streaming |
| Frontend | Static HTML + vanilla JS (or HTMX), libraries vendored | No build step, works offline. `marked` + `DOMPurify` to render model output safely |
| LLM + embeddings | Ollama HTTP API via the `ollama` Python client | `/api/chat`, `/api/embed`, `/api/tags`, `/api/show` |
| Storage | One SQLite file per workspace: tables + FTS5 (BM25) + `sqlite-vec` (vectors) | Metadata, keyword index and vectors in one transactional file, so they can't drift apart. LanceDB if a corpus ever outgrows it |
| PDF parsing | PyMuPDF / `pymupdf4llm` | Fast, keeps page numbers for citations. AGPL, fine for personal use. Docling or GROBID later for scholarly layout and metadata |
| Other formats | Native `.md`/`.txt`; `python-docx`; `ebooklib` (EPUB); `trafilatura` (HTML) | Added incrementally |
| Token budgeting | Character-based estimate with a safety margin | Exact counts aren't needed |
| File watching (later) | `watchfiles` | Re-index while the server runs |

Phase 0 risk: `sqlite-vec` needs `sqlite3.enable_load_extension`, which some Python builds
disable. Homebrew and uv-managed Pythons normally allow it; the `sqlite-vec` pip package
bundles the extension itself.

### 4. Workspace model & on-disk layout

The directory `localrag` runs in is the workspace. Like git, it walks up from the CWD to
find `.localrag/`; `localrag init` creates one.

```
my-research-topic/            ← workspace root = corpus root
├── papers/…pdf               ← any supported files, recursive
├── notes.md
├── .localragignore           ← optional, gitignore syntax
└── .localrag/
    ├── config.toml           ← embed model, chunking params, default chat model, port
    ├── index.db              ← SQLite: documents, chunks, FTS5, vectors, chat sessions
    ├── eval.jsonl            ← optional gold questions for this corpus (§10)
    └── logs/
```

- Global defaults live in `~/.config/localrag/config.toml` (Ollama host, default models);
  workspace config overrides them.
- Isolation: index, embeddings, chat sessions and the selected model all live in
  `.localrag/`. Deleting that folder fully resets the workspace.
- Always ignored: `.localrag/`, `.git/`, hidden files, `node_modules/`, files over a size cap.
- Nested workspaces: the nearest `.localrag/` wins, and a parent workspace skips any
  subfolder that has its own `.localrag/`.

### 5. Ingestion & incremental indexing (R5)

```
scan → diff against DB → parse → chunk → embed → write   (one transaction per file)
```

**Change detection**, per file:
1. `(size, mtime_ns)` unchanged → skip (fast path, no file read).
2. Otherwise hash the file (SHA-256). Same hash (touched or copied) → update mtime, skip.
3. Hash changed → reparse and rechunk, replace the document's chunks, re-embed. Chunks whose
   `text_sha` already has a vector reuse it, so a small edit doesn't re-embed a whole book.
4. In the DB but gone from disk → delete the document and its chunks. A "new" file whose
   hash matches a "deleted" one is a rename: update the path, keep everything else.

**Index-wide invalidation.** The `meta` table records `embed_model`, `embed_dim`,
`chunker_version` and parser versions. Changing the embedding model re-embeds everything;
a new chunker version rechunks everything. Changing the *chat* model never reindexes.

**Crash safety.** Each file commits in its own transaction with a `status`
(`pending | indexed | failed`). An interrupted run resumes cleanly. Parse failures are
recorded with the error and retried only when the file changes.

**Startup.** `localrag serve` syncs first (target: under a second for a few hundred
unchanged files), then serves. A large first index runs as a background task with progress
in the UI; search works over whatever is already indexed.

**Parsing** yields `Block(text, page, section_path)` so every chunk knows its pages and
heading. PDFs with no text layer are flagged *needs OCR* instead of being indexed as empty.
Metadata (title, authors, year) comes from PDF metadata, then first-page heuristics, later
a `.bib` file or Zotero.

**Chunking** is structure-aware: split on headings, then paragraphs; pack to ~500 tokens
(configurable) with ~15% overlap; never split mid-sentence. Each chunk records its document,
order, page range, section and character offsets, so neighbors can be fetched for context.

**Embedding** batches calls to `/api/embed`. Per-model templates handle required prefixes
(for example `nomic-embed-text` expects `search_document:` / `search_query:`;
`qwen3-embedding` takes an instruction on queries).

### 6. Storage schema (sketch)

```sql
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
  -- schema_version, embed_model, embed_dim, chunker_version

CREATE TABLE documents (
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE NOT NULL,            -- relative to workspace root
  size INTEGER, mtime_ns INTEGER, sha256 TEXT,
  title TEXT, authors TEXT, year INTEGER,
  parser TEXT, parser_version TEXT,
  status TEXT, error TEXT, indexed_at TEXT
);

CREATE TABLE chunks (
  id INTEGER PRIMARY KEY,
  doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
  ord INTEGER, text TEXT, text_sha TEXT,
  page_start INTEGER, page_end INTEGER, section TEXT,
  char_start INTEGER, char_end INTEGER, n_tokens INTEGER
);

CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', content_rowid='id');
CREATE VIRTUAL TABLE chunks_vec USING vec0(embedding float[1024]);  -- rowid = chunks.id; dim from meta

CREATE TABLE sessions (id INTEGER PRIMARY KEY, title TEXT, model TEXT, created_at TEXT);
CREATE TABLE messages (
  id INTEGER PRIMARY KEY, session_id INTEGER REFERENCES sessions(id),
  role TEXT, content TEXT, citations_json TEXT, supported INTEGER, created_at TEXT
);
```

Virtual tables don't cascade, so deleting a document removes its FTS and vector rows
explicitly in the same transaction.

### 7. Retrieval

1. **Standalone query (follow-ups only).** With chat history, the chat model first rewrites
   the question ("what about its limitations?" → "limitations of method X").
2. **Hybrid search.** Top ~30 from BM25 (FTS5) and top ~30 from vector kNN, merged with
   Reciprocal Rank Fusion. BM25 catches exact names, acronyms and terms; vectors catch
   paraphrase.
3. **Diversify.** Cap chunks per document (or MMR) so one long book doesn't crowd out
   everything else.
4. **Expand.** Optionally attach neighboring chunks (`ord ± 1`) to top hits.
5. **Budget.** Take top chunks until the context budget is full (§8, `num_ctx`); usually 6–10.
6. **Rerank (later).** LLM or cross-encoder reranker, added only if the eval shows a gain.

The **Search** tab exposes steps 2–3 directly: ranked chunks with document, page and
highlighted terms, no LLM involved.

Brute-force vector search is fine into the tens of thousands of chunks (roughly 500 papers).
Past that, use sqlite-vec's quantized vectors or move to LanceDB behind the same interface.

### 8. Answer generation & grounding (R4)

Layered defenses, cheapest first:

1. **Retrieval gate.** If no chunk clears a relevance threshold (calibrated on the eval
   set), skip generation and go straight to the "not found" response. Faster, and nothing
   to hallucinate from.
2. **Prompt contract.** Sources are numbered blocks (`[S3] Smith 2021, "Title", pp. 12–13`).
   The system prompt says: use only these sources; cite every factual sentence with `[S#]`;
   if they don't support an answer, begin with a fixed sentinel (`INSUFFICIENT:`) and say
   what is missing.
3. **Citation validation** (post-processing). Parse markers; flag markers that point at
   nonexistent sources and sentences with no citation. Zero valid citations → treated as
   unsupported. The UI shows uncited sentences with a warning style rather than hiding them.
4. **Strict mode** (optional, slower). A second pass checks each cited sentence against its
   chunk ("does this passage support the claim? yes / partial / no") and flags or removes
   failures. Off by default on this hardware; useful for questions that matter.

**The "I don't know" response** includes:
- A plain statement that the corpus doesn't support an answer.
- The closest material that *was* found and why it falls short, so the user can judge.
- Suggested material to add: topics, keywords, kinds of sources. Specific titles from the
  model's own memory are labeled *unverified*, since local models invent plausible papers.
  A grounded version (Phase 5): mine the **reference lists of already-indexed papers**
  ("Paper A cites Jones 2019 on exactly this; consider adding it").

**Ollama details that matter:**
- Set `num_ctx` explicitly on every request (8k–16k). Ollama's default window can be
  smaller than a RAG prompt and overflow is truncated silently, the classic local-RAG failure.
- Thinking models (`qwen3.5`, `deepseek-r1`): pass `think`, show the reasoning in a
  collapsible block, and validate citations on the answer only.
- Low temperature (0.1–0.3).
- Model dropdown: `/api/tags` filtered to models whose `/api/show` capabilities include
  `completion`, which hides embedding-only models.

**Return type.** The core returns a structured object; the web UI, the CLI and the future
reasoning agent all consume it:

```python
@dataclass
class Answer:
    text: str                       # markdown with [S#] markers
    supported: bool                 # False → "I don't know" path
    citations: list[Citation]       # marker → chunk_id, doc, pages, quoted span
    sources: list[RetrievedChunk]
    unsupported_sentences: list[str]
    suggestions: list[str]
    model: str
    thinking: str | None
```

### 9. Server, API & frontend

Bind to `127.0.0.1` only, since the server can read your files. Pick a free port
automatically and open the browser on start.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/status` | workspace path, doc/chunk counts, indexing progress, Ollama reachable |
| `GET /api/models` | installed chat-capable models + workspace default |
| `POST /api/search` | retrieval only → ranked chunks |
| `POST /api/ask` | SSE: `sources` → `thinking` / `token` events → final `answer` with validated citations |
| `GET /api/documents[/{id}]` | library browsing |
| `GET /api/chunks/{id}` | chunk text + neighbors |
| `GET /files/{doc_id}` | original file, looked up by ID only (no arbitrary paths); PDFs open at `#page=N` |
| `POST /api/reindex` | trigger a sync |
| `/api/sessions…` | per-workspace conversation history |

One page, three tabs:
- **Ask:** chat with the model dropdown. Citation chips like `[S3]` open a side panel with
  the chunk text and "Open PDF at p. 12". Unsupported answers look visibly different.
- **Search:** raw retrieval results, filterable by document or year.
- **Library:** indexed documents with status (indexed / failed / needs OCR) and a reindex button.

Render model output through `marked` + `DOMPurify`: document text can contain HTML or
injected instructions, and it reaches the page through the model.

### 10. Testing & evaluation

- **Unit tests:** change detection (touch, edit, rename, delete, embed-model switch),
  chunker page/section bookkeeping, citation parser and validator, ignore rules.
- **Fixture corpus:** a few small public-domain / arXiv PDFs and markdown files in
  `tests/fixtures/`.
- **Retrieval eval:** `eval.jsonl` rows of `{question, expected_doc, expected_pages}`;
  report recall@5 and recall@10. Use it to tune chunk size, k, fusion weights and the gate.
- **Grounding eval:** include *unanswerable* questions. Measure the refusal rate on those
  and citation precision on answerable ones. This is how R4 gets measured, and how to
  compare `qwen3.5:4b` vs `9b` vs others on your own corpus.
- `localrag eval` runs both against the current workspace.

### 11. Code layout

```
localRAG/
├── pyproject.toml
├── src/localrag/
│   ├── cli.py            # init | index | serve | search | ask | status | doctor | eval
│   ├── config.py         # global + workspace config, merged
│   ├── workspace.py      # find .localrag/ walking up; ignore rules
│   ├── db.py             # connection, sqlite-vec loading, schema, migrations
│   ├── ingest/
│   │   ├── scanner.py    # walk, fingerprint, diff
│   │   ├── parsers/      # pdf, markdown, text, docx, html, epub
│   │   ├── chunker.py
│   │   └── pipeline.py   # orchestration, transactions, progress
│   ├── embed.py
│   ├── retrieval.py      # bm25, vector, fusion, diversity, expansion, budgeting
│   ├── llm.py            # Ollama wrapper: model list, streaming chat, num_ctx, think
│   ├── prompts.py
│   ├── answer.py         # RagEngine.ask(): gate → generate → validate → Answer
│   ├── server/           # FastAPI app, routes, SSE
│   └── web/              # index.html, app.js, style.css, vendor/
└── tests/
```

Rule: `retrieval.py` and `answer.py` know nothing about HTTP. The server and CLI are thin
shells over `RagEngine`, which is what makes the reasoning agent (§13) cheap to add.

### 12. Milestones

Each phase ends in something usable.

**Phase 0: spikes & scaffold**
- `git init`; `uv init --package`; settle the Python version.
- Pull an embedding model; confirm `/api/embed` and `/api/chat` (with explicit `num_ctx`)
  from Python.
- Confirm `sqlite-vec` loads and FTS5 is available.
- *Exit:* `localrag doctor` checks all of the above and reports green.

**Phase 1: workspace + stateful indexing (CLI)**
- `init`, config, workspace discovery, ignore rules.
- Scanner and diff, PDF/MD/TXT parsers, chunker, embedder, DB writes.
- `localrag index`, `localrag status`.
- *Exit:* indexing a folder twice → second run reports 0 changed in about a second; editing,
  deleting or renaming one file reprocesses only that file. Covered by tests.

**Phase 2: retrieval + grounded answers (CLI)**
- Hybrid search, fusion, diversity, budgeting → `localrag search "…"`.
- Prompting, streaming, citation validation, "I don't know" path → `localrag ask "…"`.
- First eval set and `localrag eval`.
- *Exit:* answers cite the right pages on the eval set; record a baseline refusal rate on
  unanswerable questions.

**Phase 3: web app**
- FastAPI, SSE, model dropdown, Ask / Search / Library tabs, citation side panel,
  open-PDF-at-page.
- `localrag serve` = sync + serve + open browser.
- *Exit:* the full loop works in the browser in two different workspaces with no state
  leaking between them.

**Phase 4: quality & polish**
- Conversation sessions and follow-up rewriting.
- Background indexing with progress; file watcher.
- DOCX, EPUB, HTML; better metadata; needs-OCR detection.
- Strict mode; tune the gate and k against the eval.

**Phase 5: extensions**
- Reasoning agent (§13).
- Reference-list mining for grounded "add this paper" suggestions.
- Zotero / BibTeX metadata (Zotero is installed on this machine).
- Reranker; OCR via `ocrmypdf`; Docling or GROBID for scholarly PDFs.
- An MCP server exposing `search` / `ask`, so other agents (Claude Code, for example) can
  query a workspace's corpus.

### 13. Future: reasoning agent (design now, build later)

A reasoning model (`deepseek-r1:8b`, or `qwen3.5:9b` with thinking) gets one tool,
`ask_corpus(question) -> Answer`, through Ollama tool calling (or a simple text protocol
for models without tool support).

```
user question
  → reasoner plans sub-questions
  → loop (max N steps): ask_corpus(sub_q) → Answer{text, supported, citations}
  → reasoner writes the final answer, citing only citations collected in the loop
  → validation: every citation maps to a chunk returned by some ask_corpus call
```

Decisions to make now so this stays cheap:
- `RagEngine.ask()` returns the structured `Answer`, including `supported`, so the agent
  knows when a sub-answer has no backing.
- Internally, citations are stable chunk IDs; `[S#]` is display numbering only. That lets
  the agent merge citations from many sub-answers.
- Keep the SSE protocol open-ended so `step`, `tool_call` and `tool_result` events can be
  added without breaking the UI.
- On 16 GB, the reasoner and the answering model should be the *same* model, or Ollama
  will swap models in and out at every step. Alternative: let the agent call `search`
  (retrieval only) and read chunks itself: fewer model calls, more context.

### 14. Open questions

1. **Corpus contents:** mostly PDFs (papers, books)? Any scanned PDFs, EPUBs, Word files,
   web clippings?
2. **Corpus size:** tens, hundreds or thousands of documents per workspace?
3. **Corpus location:** is the workspace folder itself the corpus, or should config also
   list outside source folders (Zotero's storage folder, for example)?
4. **Zotero:** is that where your literature lives? Its metadata (authors, year, tags,
   collections) could drive citations, and collections could map to workspaces.
5. **Chat history:** keep conversations per workspace, or treat every question as standalone?
6. **Citation granularity:** is `[Author Year, p. N]` enough, or do you want exact quoted
   spans highlighted in the source?
7. **Language:** is Python acceptable? It has the strongest parsing ecosystem; a Rust or Go
   single binary is possible later but would cost a lot of PDF-parsing quality.
