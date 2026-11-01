// localrag web UI (ARCHITECTURE.md §15). Plain ES module, no build step.

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...children) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null && v !== false) e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c !== null && c !== undefined) e.append(c);
  return e;
};
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function api(path, opts = {}) {
  const res = await fetch(path, {
    ...opts,
    credentials: "same-origin",
    headers: opts.body ? { "Content-Type": "application/json", ...(opts.headers || {}) } : opts.headers,
  });
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    if (body?.error === "no_workspace") showWorkspaceDialog();
    throw Object.assign(new Error(body?.message || body || res.statusText), { status: res.status, body });
  }
  return body;
}

const STATUS_TEXT = {
  verified: "✓ Verified quote",
  verified_paraphrase: "≈ Verified paraphrase",
  unverified: "⚠ Unverified",
};
const STATUS_ICON = { verified: "✓", verified_paraphrase: "≈", unverified: "⚠" };

const state = {
  sessionId: null,
  // Message id → { citations, supported, el } for citations shown in the thread.
  messages: new Map(),
  inspected: null,
};

// ---------------------------------------------------------------- tabs & status

document.querySelectorAll(".tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.id === `tab-${b.dataset.tab}`));
    if (b.dataset.tab === "documents") loadDocuments();
    if (b.dataset.tab === "search") loadSearchDocs();
  }),
);

let syncWasRunning = false;
let statusTimer = null;
async function refreshStatus() {
  clearTimeout(statusTimer);
  try {
    const s = await api("/api/status");
    if (!s.workspace) {
      $("#ws-btn-name").textContent = "No workspace";
      $("#status").innerHTML = `<span class="dot bad"></span>No workspace open`;
      return;
    }
    $("#ws-btn-name").textContent = s.name || "Workspace";
    document.title = s.name ? `localRAG · ${s.name}` : "localRAG";
    const docs = Object.values(s.documents || {}).reduce((a, b) => a + b, 0);
    let dot = "", text;
    if (!s.ollama.reachable) {
      dot = "bad";
      text = `Ollama not reachable at ${s.ollama.host}`;
    } else if (s.sync.running) {
      dot = "busy";
      const o = s.sync.ocr;
      text = s.sync.phase === "ocr" && o ? `OCR: ${o.path} · page ${Math.min(o.done + 1, o.pages)} of ${o.pages}`
        : s.sync.phase === "metadata" ? `Looking up citations on Crossref…${s.sync.current ? ` · ${s.sync.current}` : ""}`
        : `Indexing… ${s.sync.processed} file(s)${s.sync.current ? ` · ${s.sync.current}` : ""}`;
    } else if (s.sync.error) {
      dot = "bad";
      text = `Indexing failed: ${s.sync.error}`;
    } else {
      text = `${docs} documents · ${s.chunks} passages`;
    }
    $("#status").innerHTML = `<span class="dot ${dot}"></span>${esc(text)}`;
    $("#status").title = s.workspace;
    $("#ocr-cancel").hidden = !(s.sync.running && s.sync.phase === "ocr");
    if (syncWasRunning && !s.sync.running && $("#tab-documents").classList.contains("active")) loadDocuments();
    syncWasRunning = s.sync.running;
    statusTimer = setTimeout(refreshStatus, s.sync.running ? 1500 : 8000);
  } catch (e) {
    $("#status").innerHTML = `<span class="dot bad"></span>${esc(e.status ? e.message : "Server unreachable")}`;
    statusTimer = setTimeout(refreshStatus, 5000);
  }
}

// ---------------------------------------------------------------- models

async function loadModels() {
  const sel = $("#model");
  try {
    const m = await api("/api/models");
    const saved = localStorageGet("model");
    sel.replaceChildren(...m.models.map((x) => el("option", { value: x.name }, x.name)));
    const pick = [saved, m.default].find((n) => m.models.some((x) => x.name === n));
    if (pick) sel.value = pick;
  } catch (e) {
    sel.replaceChildren(el("option", { value: "" }, "(Ollama unavailable)"));
  }
}
$("#model").addEventListener("change", (e) => localStorageSet("model", e.target.value));

function localStorageGet(k) {
  try { return localStorage.getItem(`localrag.${k}`); } catch { return null; }
}
function localStorageSet(k, v) {
  try { localStorage.setItem(`localrag.${k}`, v); } catch { /* private mode */ }
}

// ---------------------------------------------------------------- sessions

async function loadSessions() {
  const { sessions } = await api("/api/sessions");
  $("#session-list").replaceChildren(
    ...sessions.map((s) =>
      el("li", { class: s.id === state.sessionId ? "active" : "", onclick: () => openSession(s.id) },
        el("span", { class: "title", title: s.title }, s.title || `Chat ${s.id}`),
        el("button", {
          class: "del", title: "Delete chat",
          onclick: async (ev) => {
            ev.stopPropagation();
            if (!confirm("Delete this chat? Its questions and answers are removed from history.")) return;
            await api(`/api/sessions/${s.id}`, { method: "DELETE" });
            if (state.sessionId === s.id) newChat();
            loadSessions();
          },
        }, "×"),
      ),
    ),
  );
}

function newChat() {
  state.sessionId = null;
  state.messages.clear();
  $("#thread").replaceChildren($("#thread-empty") || emptyState());
  $("#thread-empty").hidden = false;
  closeInspector();
  loadSessions();
  $("#question").focus();
}
function emptyState() {
  return el("div", { class: "empty", id: "thread-empty" },
    el("h2", {}, "Ask your literature"),
    el("p", {}, "Answers come only from the documents in corpus/, with citations you can check against the source."));
}
$("#new-chat").addEventListener("click", newChat);

async function openSession(id) {
  const { messages } = await api(`/api/sessions/${id}`);
  state.sessionId = id;
  state.messages.clear();
  closeInspector();
  const thread = $("#thread");
  thread.replaceChildren();
  for (const m of messages) {
    if (m.role === "user") thread.append(userMsg(m.content));
    else thread.append(assistantFromHistory(m));
  }
  thread.scrollTop = thread.scrollHeight;
  loadSessions();
}

// ---------------------------------------------------------------- rendering answers

function userMsg(text) {
  return el("div", { class: "msg user" }, text);
}

/// Normalizes live-answer citations and history snapshots into one shape.
function liveCitation(c) {
  return {
    ordinal: c.ordinal, status: c.status, ocr: c.chunk.ocr, label: c.chunk.locator ? `${c.chunk.doc_label}, ${c.chunk.locator}` : c.chunk.doc_label,
    quote: c.quote, span: c.quote_verified ? c.span : null, quote_repaired: c.quote_repaired, support: c.support,
    chunk_id: c.chunk.id, doc_sha256: c.chunk.doc_sha256, chunk_text_sha: c.chunk.text_sha,
  };
}
function snapshotCitation(c) {
  return {
    ordinal: c.ordinal, status: c.status || (c.quote_verified ? "verified" : "unverified"), label: c.label,
    quote: c.quote, span: c.span, chunk_id: c.chunk_id, doc_sha256: c.doc_sha256, chunk_text_sha: c.chunk_text_sha,
    quote_repaired: c.quote_repaired, support: c.support,
  };
}

function renderMarkdownWithChips(placeholderMd, citations, messageId) {
  const byOrd = new Map(citations.map((c) => [c.ordinal, c]));
  const html = DOMPurify.sanitize(marked.parse(placeholderMd || ""));
  return html.replace(/⟦(\d+)⟧/g, (_, n) => {
    const c = byOrd.get(Number(n));
    if (!c) return `<span class="chip invalid" title="This citation names a source that wasn't provided">invalid</span>`;
    const ocr = c.ocr ? `<span class="ocr-badge" title="From OCR of a scanned page">OCR</span>` : "";
    const rejected = c.support === "no";
    const title = rejected ? "Strict check: the passage does not support this sentence" : STATUS_TEXT[c.status];
    return `<button class="chip ${esc(rejected ? "unverified" : c.status)}" data-msg="${esc(messageId ?? "")}" data-ord="${c.ordinal}" title="${esc(title)}">${esc(c.label)} ${rejected ? "✗" : STATUS_ICON[c.status] || ""}${ocr}</button>`;
  });
}

function bindChips(container, messageId) {
  container.querySelectorAll("button.chip").forEach((b) =>
    b.addEventListener("click", () => {
      const entry = state.messages.get(messageId);
      const c = entry?.citations.find((x) => x.ordinal === Number(b.dataset.ord));
      if (c) inspectCitation(c, messageId, b);
    }),
  );
}

function answerBlock({ placeholderMd, text, citations, supported, notFound, sources, suggestions, warnings, unsupported, invalid, thinking, model, messageId }) {
  const wrap = el("div", { class: "msg assistant" });
  for (const w of warnings || []) wrap.append(el("div", { class: "warning" }, "⚠ ", w.message || describeWarning(w)));
  if (thinking) wrap.append(el("details", {}, el("summary", {}, "Model reasoning"), el("div", { class: "thinking" }, thinking)));

  const insufficient = notFound || (text || "").startsWith("INSUFFICIENT");
  const body = el("div", { class: "body" });
  if (insufficient) {
    body.append(el("p", {}, notFound ? text : text.replace(/^INSUFFICIENT:\s*/, "The sources don't contain enough to answer: ")));
  } else {
    body.innerHTML = renderMarkdownWithChips(placeholderMd, citations, messageId);
  }
  wrap.append(body);

  if (insufficient && sources?.length) {
    wrap.append(el("div", { class: "closest" }, el("div", { class: "muted" }, "Closest material found (not sufficient to answer):"),
      el("ul", {}, sources.map((c) => el("li", {}, el("button", { class: "chip unverified", onclick: () => inspectChunk(c) }, `${c.doc_label}${c.locator ? `, ${c.locator}` : ""}`))))));
  }
  if (suggestions?.length) {
    wrap.append(el("div", { class: "suggestions" },
      el("div", { class: "label" }, "Material that could help (suggested from the model's general knowledge, unverified):"),
      el("ul", {}, suggestions.map((s) => el("li", {}, s)))));
  }
  const flags = [];
  if (!insufficient && supported === false) flags.push(el("span", { class: "badge unsupported" }, "Not supported by a verified citation"));
  if (unsupported?.length) flags.push(el("details", {}, el("summary", {}, `${unsupported.length} sentence(s) without a citation`), el("ul", {}, unsupported.map((s) => el("li", {}, s)))));
  if (invalid?.length) flags.push(el("div", {}, `Citations to sources that weren't provided: ${invalid.join(", ")}`));
  if (flags.length) wrap.append(el("div", { class: "flags" }, flags));
  if (model) wrap.append(el("div", { class: "meta" }, model));
  if (messageId !== undefined) {
    state.messages.set(messageId, { citations, supported, el: wrap });
    bindChips(body, messageId);
  }
  return wrap;
}

function describeWarning(w) {
  return w.kind ? w.kind.replace(/_/g, " ") : "Context warning";
}

function assistantFromHistory(m) {
  const x = m.extra || {};
  return answerBlock({
    placeholderMd: m.placeholder_markdown,
    text: m.content,
    citations: (m.citations || []).map(snapshotCitation),
    supported: m.supported,
    notFound: x.not_found ?? m.content.startsWith("The indexed literature doesn't contain"),
    suggestions: x.suggestions,
    unsupported: x.unsupported_sentences,
    invalid: x.invalid_markers,
    sources: x.closest,
    warnings: Array.isArray(m.warnings) ? m.warnings.map((w) => ({ ...w, message: warningText(w) })) : [],
    model: m.model,
    messageId: m.id,
  });
}

function warningText(w) {
  switch (w.kind) {
    case "history_dropped": return `Context full: ${w.turns} earliest turn(s) of this chat (${w.tokens} tokens) were left out.`;
    case "sources_dropped": return `${w.dropped} of ${w.relevant} relevant passages didn't fit; the answer used the top ${w.relevant - w.dropped}.`;
    case "neighbors_dropped": return `Surrounding context for ${w.passages} passages was left out.`;
    case "truncated": return `The model received less than was sent (${w.evaluated} of ~${w.sent} tokens); the answer may be missing information.`;
    default: return describeWarning(w);
  }
}

// ---------------------------------------------------------------- asking

$("#question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) $("#composer").requestSubmit();
});
$("#question").addEventListener("input", () => ($("#ask-error").textContent = ""));

$("#composer").addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = $("#question").value.trim();
  if (!question) return;
  const btn = $("#ask-btn");
  btn.disabled = true;
  $("#ask-error").textContent = "";
  $("#thread-empty")?.setAttribute("hidden", "");

  let res;
  try {
    res = await fetch("/api/ask", {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, model: $("#model").value || null, session_id: state.sessionId }),
    });
  } catch (err) {
    $("#ask-error").textContent = `Couldn't reach the server: ${err.message}`;
    btn.disabled = false;
    return;
  }
  if (!res.ok) {
    // Over budget: keep the text, explain by how much (§11).
    const body = await res.json().catch(() => ({}));
    $("#ask-error").textContent = body.message || `Request failed (${res.status})`;
    btn.disabled = false;
    return;
  }

  $("#question").value = "";
  const thread = $("#thread");
  thread.append(userMsg(question));
  const live = el("div", { class: "msg assistant" });
  const warnBox = el("div");
  const thinkBox = el("div", { class: "thinking", hidden: true });
  const stream = el("div", { class: "streaming" }, "Searching the literature…");
  live.append(warnBox, thinkBox, stream);
  thread.append(live);
  thread.scrollTop = thread.scrollHeight;

  let started = false;
  try {
    for await (const { event, data } of sse(res)) {
      if (event === "query") stream.textContent = `Searching for: ${data.query}`;
      else if (event === "sources") { if (!started) stream.textContent = `Reading ${data.sources.length} passages…`; }
      else if (event === "warning") warnBox.append(el("div", { class: "warning" }, "⚠ ", data.message));
      else if (event === "thinking") { thinkBox.hidden = false; thinkBox.textContent += data.text; }
      else if (event === "token") {
        if (!started) { stream.textContent = ""; started = true; }
        stream.textContent += data.text;
        thread.scrollTop = thread.scrollHeight;
      } else if (event === "answer") {
        const a = data.answer;
        state.sessionId = data.session_id;
        live.replaceWith(answerBlock({
          placeholderMd: data.placeholder_markdown, text: a.text, citations: a.citations.map(liveCitation),
          supported: a.supported, notFound: a.not_found, sources: a.sources, suggestions: a.suggestions,
          warnings: a.warnings.map((w) => ({ ...w, message: warningText(w) })), unsupported: a.unsupported_sentences,
          invalid: a.invalid_markers, thinking: a.thinking, model: a.model, messageId: data.message_id,
        }));
        loadSessions();
      } else if (event === "error") {
        stream.textContent = "";
        live.append(el("div", { class: "error" }, data.message));
      }
    }
  } catch (err) {
    live.append(el("div", { class: "error" }, `Connection lost: ${err.message}`));
  } finally {
    btn.disabled = false;
    thread.scrollTop = thread.scrollHeight;
  }
});

/// Server-sent events from a fetch() response (EventSource can't POST).
async function* sse(res) {
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, i);
      buf = buf.slice(i + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) yield { event, data: JSON.parse(data) };
    }
  }
}

// ---------------------------------------------------------------- inspector

function closeInspector() {
  $("#inspector").hidden = true;
  document.querySelectorAll(".chip.selected").forEach((c) => c.classList.remove("selected"));
  state.inspected = null;
}
$("#insp-close").addEventListener("click", closeInspector);
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!$("#viewer").hidden) closeViewer();
  else if (!$("#meta-dialog").hidden) $("#meta-dialog").hidden = true;
  else if (!$("#settings-dialog").hidden) $("#settings-dialog").hidden = true;
  else closeInspector();
});

/// Chunk text with a highlight: the verified quote span, or (§10) the whole chunk when the
/// citation is unverified or a verified paraphrase.
function renderChunkText(text, span, mode) {
  const box = $("#insp-text");
  box.className = `insp-text${mode === "whole" || mode === "para" ? " whole" : ""}${mode === "para" ? " para" : ""}`;
  const chars = Array.from(text);
  if (mode === "none") { box.textContent = text; return; }
  const [a, b] = mode === "quote" && span ? span : [0, chars.length];
  box.replaceChildren(chars.slice(0, a).join(""), el("mark", {}, chars.slice(a, b).join("")), chars.slice(b).join(""));
}

async function inspectCitation(citation, messageId, chipEl) {
  let c = citation;
  document.querySelectorAll(".chip.selected").forEach((x) => x.classList.remove("selected"));
  chipEl?.classList.add("selected");
  const insp = $("#inspector");
  insp.hidden = false;
  $("#insp-label").textContent = c.label;
  $("#insp-status").innerHTML = `<span class="chip ${esc(c.status)}">${esc(STATUS_TEXT[c.status])}</span>`;
  const q = $("#insp-quote");
  q.hidden = !c.quote;
  if (c.quote) {
    q.replaceChildren(el("span", { class: "muted" }, c.status === "verified" ? "Quoted: " : "Model's quote (not found verbatim): "), el("span", { class: "q" }, `“${c.quote}”`));
  }
  $("#insp-text").textContent = "Loading passage…";
  const note = $("#insp-note");
  note.hidden = true;

  let chunk;
  try {
    const qs = new URLSearchParams({ text_sha: c.chunk_text_sha, doc_sha: c.doc_sha256 || "" });
    if (c.status === "verified" && c.quote) qs.set("quote", c.quote);
    const r = await api(`/api/chunks/${c.chunk_id}?${qs}`);
    chunk = r.chunk;
    // The span stored with the citation can be missing (older history) or stale (the
    // passage was relocated); the server's fresh match wins.
    if (r.span) c = { ...c, span: r.span };
    if (r.resolution === "relocated") showNote("The document changed since this answer; showing the same passage in the current version.");
    if (r.resolution === "source_removed") showNote("This document was removed from corpus/. The passage is kept for this citation.");
  } catch (e) {
    $("#insp-text").textContent = "";
    showNote(e.status === 404 ? "This passage is no longer in the index (the document was forgotten or changed)." : e.message);
    $("#insp-open").hidden = true;
    $("#insp-paraphrase").hidden = true;
    return;
  }
  const mode = c.status === "verified" ? "quote" : c.status === "verified_paraphrase" ? "para" : "whole";
  renderChunkText(chunk.text, c.span, mode);
  if (c.quote_repaired) showNote("The model's first quote wasn't in the source, so the exact supporting words were re-selected from this passage afterwards and verified.");
  if (c.support) {
    const text = { yes: "Strict check: the passage supports this sentence.", partial: "Strict check: the passage supports only part of this sentence.", no: "Strict check: the passage does NOT support this sentence." }[c.support];
    $("#insp-status").append(" ", el("span", { class: `chip ${c.support === "yes" ? "verified" : "unverified"}` }, text));
  }
  if (chunk.ocr) $("#insp-label").append(el("span", { class: "ocr-badge", title: "Text recognized from a scanned page; small recognition errors are possible" }, "OCR"));
  if (mode !== "quote") {
    showNote(c.status === "unverified"
      ? "The quote wasn't found word for word, so the whole cited passage is highlighted. Compare it with the claim; if it supports it, mark it as a verified paraphrase."
      : "You reviewed this citation and accepted it as a paraphrase of the highlighted passage.");
  }
  state.inspected = { chunk, span: mode === "quote" ? c.span : null, mode };
  setupOpenButton(chunk, mode === "quote" ? c.span : null, mode);

  const pb = $("#insp-paraphrase");
  pb.hidden = !(messageId && (c.status === "unverified" || c.status === "verified_paraphrase"));
  pb.textContent = c.status === "unverified" ? "Mark as verified paraphrase" : "Undo verified paraphrase";
  pb.onclick = async () => {
    pb.disabled = true;
    try {
      const { message } = await api(`/api/messages/${messageId}/citations/${c.ordinal}/paraphrase`, {
        method: "POST", body: JSON.stringify({ on: c.status === "unverified" }),
      });
      applyMessageUpdate(messageId, message);
      const updated = state.messages.get(messageId).citations.find((x) => x.ordinal === c.ordinal);
      const chip = state.messages.get(messageId).el.querySelector(`.chip[data-ord="${c.ordinal}"]`);
      inspectCitation(updated, messageId, chip);
    } catch (e) {
      showNote(e.message);
    } finally {
      pb.disabled = false;
    }
  };
}

function applyMessageUpdate(messageId, m) {
  const entry = state.messages.get(messageId);
  if (!entry) return;
  const fresh = assistantFromHistory(m);
  entry.el.replaceWith(fresh);
}

async function inspectChunk(chunk) {
  closeInspector();
  $("#inspector").hidden = false;
  $("#insp-label").textContent = chunk.locator ? `${chunk.doc_label}, ${chunk.locator}` : chunk.doc_label;
  $("#insp-status").textContent = chunk.doc_path;
  $("#insp-status").className = "muted";
  $("#insp-quote").hidden = true;
  $("#insp-note").hidden = true;
  $("#insp-paraphrase").hidden = true;
  renderChunkText(chunk.text, null, "none");
  setupOpenButton(chunk, null, "whole");
}

function showNote(text) {
  const n = $("#insp-note");
  n.textContent = text;
  n.hidden = false;
}

function setupOpenButton(chunk, span, mode) {
  const b = $("#insp-open");
  b.hidden = false;
  b.textContent = chunk.locator?.startsWith("p") ? `Open at ${chunk.locator}` : "Open in source";
  b.onclick = () => openViewer(chunk, span, mode);
}

// ---------------------------------------------------------------- source viewer

let pdfjs = null;
const viewer = { doc: null, page: 1, rects: new Map(), cls: "quote", docId: null };

async function openViewer(chunk, span, mode) {
  const qs = new URLSearchParams({ chunk: chunk.id });
  if (span) { qs.set("start", span[0]); qs.set("end", span[1]); }
  const hl = await api(`/api/highlights?${qs}`);
  $("#viewer-title").textContent = chunk.locator ? `${chunk.doc_label}, ${chunk.locator}` : chunk.doc_label;
  $("#viewer").hidden = false;
  const body = $("#viewer-body");
  body.replaceChildren(el("div", { class: "muted" }, "Loading…"));
  viewer.cls = mode === "quote" ? "quote" : mode === "para" ? "para" : "whole";

  if (hl.kind === "unavailable") {
    body.replaceChildren(el("div", { class: "text-view" }, `The source can't be shown: ${hl.reason}.`));
    toggleNav(false);
    return;
  }
  if (hl.kind === "text") {
    toggleNav(false);
    const text = await (await fetch(`/files/${chunk.doc_id}`, { credentials: "same-origin" })).text();
    const chars = Array.from(text);
    const mark = el("mark", {}, chars.slice(hl.start, hl.end).join(""));
    const view = el("div", { class: `text-view${viewer.cls === "quote" ? "" : " whole"}${viewer.cls === "para" ? " para" : ""}` },
      chars.slice(0, hl.start).join(""), mark, chars.slice(hl.end).join(""));
    body.replaceChildren(view);
    mark.scrollIntoView({ block: "center" });
    return;
  }

  // PDF
  if (!pdfjs) {
    pdfjs = await import("/assets/vendor/pdf.min.mjs");
    pdfjs.GlobalWorkerOptions.workerSrc = "/assets/vendor/pdf.worker.min.mjs";
  }
  if (viewer.docId !== chunk.doc_id) {
    // isEvalSupported: false keeps PDF.js from compiling font code from the PDF (defense in
    // depth against malicious documents, cf. CVE-2024-4367).
    viewer.doc = await pdfjs.getDocument({ url: `/files/${chunk.doc_id}`, withCredentials: true, isEvalSupported: false }).promise;
    viewer.docId = chunk.doc_id;
  }
  viewer.rects = new Map(hl.pages.map((p) => [p.pdf_page, p.rects]));
  viewer.page = hl.pages[0]?.pdf_page || chunk.pdf_page_start || 1;
  toggleNav(true);
  renderPage();
}

async function renderPage() {
  const body = $("#viewer-body");
  const page = await viewer.doc.getPage(viewer.page);
  const base = page.getViewport({ scale: 1 });
  const width = Math.min(body.clientWidth - 40, 980);
  const scale = width / base.width;
  const vp = page.getViewport({ scale: scale * window.devicePixelRatio });
  const canvas = el("canvas");
  canvas.width = vp.width;
  canvas.height = vp.height;
  canvas.style.width = `${vp.width / window.devicePixelRatio}px`;
  canvas.style.height = `${vp.height / window.devicePixelRatio}px`;
  const wrap = el("div", { class: "page-wrap" }, canvas);
  body.replaceChildren(wrap);
  await page.render({ canvasContext: canvas.getContext("2d"), viewport: vp }).promise;
  const cssW = vp.width / window.devicePixelRatio, cssH = vp.height / window.devicePixelRatio;
  // Scroll to the topmost rectangle: PDF text order isn't reading order (footers come first).
  let top = null, topY = Infinity;
  for (const [x, y, w, h] of viewer.rects.get(viewer.page) || []) {
    const r = el("div", { class: `hl ${viewer.cls}` });
    Object.assign(r.style, { left: `${x * cssW}px`, top: `${y * cssH}px`, width: `${w * cssW}px`, height: `${h * cssH}px` });
    wrap.append(r);
    if (y < topY) { topY = y; top = r; }
  }
  top?.scrollIntoView({ block: "start" });
  $("#viewer-page").textContent = `Page ${viewer.page} of ${viewer.doc.numPages}`;
  $("#viewer-prev").disabled = viewer.page <= 1;
  $("#viewer-next").disabled = viewer.page >= viewer.doc.numPages;
}

function toggleNav(on) {
  for (const id of ["#viewer-prev", "#viewer-next", "#viewer-page"]) $(id).hidden = !on;
}
$("#viewer-prev").addEventListener("click", () => { viewer.page--; renderPage(); });
$("#viewer-next").addEventListener("click", () => { viewer.page++; renderPage(); });
function closeViewer() { $("#viewer").hidden = true; }
$("#viewer-close").addEventListener("click", closeViewer);

// ---------------------------------------------------------------- search

$("#search-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const list = $("#search-results");
  list.replaceChildren(el("li", { class: "muted" }, "Searching…"));
  try {
    const doc = $("#search-doc").value;
    const yf = parseInt($("#search-from").value, 10), yt = parseInt($("#search-to").value, 10);
    const filter = { doc_ids: doc ? [Number(doc)] : null, year_from: Number.isFinite(yf) ? yf : null, year_to: Number.isFinite(yt) ? yt : null };
    const { hits } = await api("/api/search", { method: "POST", body: JSON.stringify({ query: $("#search-q").value, k: 15, filter }) });
    if (!hits.length) { list.replaceChildren(el("li", { class: "muted" }, "No matches.")); return; }
    list.replaceChildren(...hits.map((h) => {
      const c = h.chunk;
      const scores = [h.similarity != null ? `similarity ${h.similarity.toFixed(2)}` : null, h.bm25_rank ? `keyword #${h.bm25_rank}` : null].filter(Boolean).join(" · ");
      return el("li", { onclick: () => inspectChunk(c) },
        el("div", { class: "head" }, c.locator ? `${c.doc_label}, ${c.locator}` : c.doc_label, el("span", { class: "scores" }, scores)),
        el("div", { class: "snippet" }, c.text.length > 320 ? `${c.text.slice(0, 320)}…` : c.text));
    }));
  } catch (err) {
    list.replaceChildren(el("li", { class: "error" }, err.message));
  }
});

async function loadSearchDocs() {
  const sel = $("#search-doc");
  const keep = sel.value;
  const { documents } = await api("/api/documents");
  sel.replaceChildren(el("option", { value: "" }, "All documents"),
    ...documents.filter((d) => d.status === "indexed" || d.status === "needs_ocr").map((d) => el("option", { value: d.id }, d.label)));
  sel.value = documents.some((d) => String(d.id) === keep) ? keep : "";
}

// ---------------------------------------------------------------- library

async function loadDocuments() {
  const { documents } = await api("/api/documents");
  const counts = documents.reduce((m, d) => ((m[d.status] = (m[d.status] || 0) + 1), m), {});
  $("#library-summary").textContent = Object.entries(counts).map(([k, n]) => `${n} ${k.replace("_", " ")}`).join(" · ") || "No documents yet: add files to corpus/ and press Reindex.";
  $("#forget-all").disabled = !counts.removed;
  $("#ocr-all").hidden = (counts.needs_ocr || 0) < 2;
  $("#library-rows").replaceChildren(...documents.map((d) => {
    const detail = d.status === "failed" ? d.error : d.status === "needs_ocr" ? `Scanned pages without text: ${(d.scanned_pages || []).join(", ")}` : null;
    const csl = d.csl || {};
    const year = csl.issued?.["date-parts"]?.[0]?.[0];
    const metaBadge = { matched: d.metadata_source === "manual" ? ["manual", "edited"] : d.metadata_source === "crossref_doi" ? ["matched", "DOI"] : ["matched", "Crossref"],
      proposed: ["proposed", "confirm"], unmatched: ["unmatched", "no match"], pending: ["pending", "pending"] }[d.metadata_status] || ["pending", "—"];
    const citeCell = el("td", { class: "cite-cell" },
      el("div", {}, d.label, el("button", { class: `meta-badge ${metaBadge[0]}`, title: "Citation metadata", onclick: () => openMetadata(d) }, metaBadge[1])),
      csl.title ? el("div", { class: "title" }, `${csl.title}${year ? ` (${year})` : ""}`) : null);
    const nameCell = el("td", {}, el("div", { class: "path" }, d.path), detail ? el("div", { class: "muted" }, detail) : null);
    return el("tr", {},
      nameCell,
      citeCell,
      el("td", {}, el("span", { class: `st ${d.status}` }, d.status.replace("_", " "))),
      el("td", {}, String(d.chunks)),
      el("td", {}, d.pages != null && d.kind === "pdf" ? String(d.pages) : "–"),
      el("td", { class: "actions" }, d.status === "removed"
        ? el("button", { class: "small danger", title: "Permanently delete this document's record, passages and OCR cache", onclick: () => forget(d) }, "Forget")
        : el("span", {},
            d.status === "needs_ocr" ? el("button", { class: "small primary", title: "Recognize the text of the scanned pages with Apple Vision", onclick: () => runOcr(d) }, "Run OCR") : null,
            " ",
            el("button", { class: "small", title: "Rename the file (no reindexing)", onclick: () => startRename(d, nameCell) }, "Rename"))));
  }));
}

/// Inline rename: Enter saves, Escape cancels. Only the name changes: the extension is shown
/// as a fixed suffix (changing the type would mean reindexing), and nothing is reindexed.
function startRename(d, cell) {
  const file = d.path.split("/").pop();
  const dot = file.lastIndexOf(".");
  const [stem, ext] = dot > 0 ? [file.slice(0, dot), file.slice(dot)] : [file, ""];
  const input = el("input", { type: "text", value: stem, class: "rename-input", spellcheck: "false" });
  const err = el("div", { class: "error" });
  const original = [...cell.childNodes];
  cell.replaceChildren(el("div", { class: "row" }, input, el("span", { class: "path" }, ext)), el("div", { class: "muted small-text" }, "Enter to save · Esc to cancel"), err);
  input.focus();
  input.select();
  input.addEventListener("keydown", async (e) => {
    if (e.key === "Escape") { cell.replaceChildren(...original); return; }
    if (e.key !== "Enter") return;
    e.preventDefault();
    let newStem = input.value.trim();
    // Typing the extension too is harmless: don't double it.
    if (ext && newStem.toLowerCase().endsWith(ext.toLowerCase())) newStem = newStem.slice(0, -ext.length).trim();
    const name = newStem + ext;
    if (!newStem || name === file) { cell.replaceChildren(...original); return; }
    input.disabled = true;
    try {
      const r = await api(`/api/documents/${d.id}/rename`, { method: "POST", body: JSON.stringify({ name }) });
      const notes = [];
      if (r.citations_updated) notes.push(`${r.citations_updated} citation(s) in chat history updated`);
      if (r.eval_items_updated) notes.push(`${r.eval_items_updated} eval question(s) updated`);
      $("#upload-results").replaceChildren(el("li", { class: "ok" }, `Renamed ${file} → ${r.path.split("/").pop()}${notes.length ? ` · ${notes.join(", ")}` : ""}. Not reindexed.`));
      loadDocuments();
    } catch (ex) {
      input.disabled = false;
      err.textContent = ex.message;
    }
  });
}

async function forget(d) {
  if (!confirm(`Permanently forget “${d.label}”?\n\nIts record, passages and OCR cache are deleted. Old answers stay readable, but their citations to it will show as forgotten. This can't be undone.`)) return;
  try { await api(`/api/documents/${d.id}`, { method: "DELETE" }); } catch (e) { alert(e.message); }
  loadDocuments();
}
$("#forget-all").addEventListener("click", async () => {
  if (!confirm("Permanently forget every removed document? This can't be undone.")) return;
  await api("/api/documents/forget-removed", { method: "POST" });
  loadDocuments();
});
$("#reindex").addEventListener("click", async () => {
  const btn = $("#reindex");
  btn.disabled = true;
  const before = (await api("/api/status")).sync.runs;
  await api("/api/reindex", { method: "POST" });
  refreshStatus();
  await waitForSync(before);
  btn.disabled = false;
  loadDocuments();
  refreshStatus();
});

// ---------------------------------------------------------------- workspaces

const wsBtn = $("#ws-btn");
const wsMenu = $("#ws-menu");
wsBtn.addEventListener("click", async (e) => {
  e.stopPropagation();
  const open = wsMenu.hidden;
  if (open) await fillRecent();
  wsMenu.hidden = !open;
  wsBtn.setAttribute("aria-expanded", String(open));
});
document.addEventListener("click", (e) => {
  if (!wsMenu.hidden && !wsMenu.contains(e.target)) { wsMenu.hidden = true; wsBtn.setAttribute("aria-expanded", "false"); }
});
$("#menu-new").addEventListener("click", () => { wsMenu.hidden = true; showWorkspaceDialog("create"); });
$("#menu-open").addEventListener("click", () => { wsMenu.hidden = true; showWorkspaceDialog("open"); });

async function fillRecent() {
  const { active, known } = await api("/api/workspaces");
  const recent = known.filter((w) => w.available).slice(0, 6);
  $("#menu-recent").replaceChildren(...(recent.length ? recent.map((w) =>
    el("button", {
      role: "menuitem", class: active?.path === w.path ? "current" : "",
      onclick: () => { wsMenu.hidden = true; if (active?.path !== w.path) switchWorkspace(w.path); },
    }, el("span", { class: "name" }, w.name), el("span", { class: "path" }, w.path)),
  ) : [el("div", { class: "menu-label" }, "None yet")]));
}

let dialogRequired = false;
async function showWorkspaceDialog(focus = "open") {
  const { active, known } = await api("/api/workspaces");
  dialogRequired = !active;
  $("#ws-close").hidden = dialogRequired;
  $("#ws-error").textContent = "";
  const usable = known.filter((w) => w.available);
  if (!known.length) {
    $("#ws-title").textContent = "Create your first workspace";
    $("#ws-intro").textContent = "A workspace is a folder holding the documents for one research topic. Choose where to create it.";
  } else {
    $("#ws-title").textContent = active ? "Workspaces" : "Choose a workspace";
    $("#ws-intro").textContent = active ? `Open: ${active.name}` : "Pick the workspace to work in, or create a new one.";
  }
  $("#ws-known-section").hidden = !known.length;
  $("#ws-known").replaceChildren(...known.map((w) =>
    el("li", { class: w.available ? "" : "missing" },
      el("div", { class: "info" },
        el("div", { class: "name" }, w.name, w.available ? null : el("span", { class: "missing-note" }, " · folder not found")),
        el("div", { class: "path", title: w.path }, w.path)),
      active?.path === w.path ? el("span", { class: "muted" }, "open")
        : el("button", { class: "small primary", disabled: !w.available, onclick: () => switchWorkspace(w.path) }, "Open"),
      active?.path === w.path ? null : el("button", {
        class: "del", title: "Remove from this list (the folder isn't touched)",
        onclick: async () => { await api("/api/workspaces/forget", { method: "POST", body: JSON.stringify({ path: w.path }) }); showWorkspaceDialog(focus); },
      }, "×"),
    )));
  updateCreateButton();
  $("#ws-dialog").hidden = false;
  if (focus === "create" || !usable.length) $("#ws-name-input").focus();
}

function closeWorkspaceDialog() {
  if (dialogRequired) return;
  $("#ws-dialog").hidden = true;
}
$("#ws-close").addEventListener("click", closeWorkspaceDialog);
$("#ws-dialog").addEventListener("keydown", (e) => { if (e.key === "Escape") { e.stopPropagation(); closeWorkspaceDialog(); } });

// The location comes only from the macOS folder chooser; it can't be typed.
$("#ws-choose").addEventListener("click", async () => {
  const r = await api("/api/workspaces/choose-folder", { method: "POST" });
  if (r.path) $("#ws-parent").value = r.path;
  updateCreateButton();
});
$("#ws-parent").addEventListener("click", () => $("#ws-choose").click());
$("#ws-name-input").addEventListener("input", updateCreateButton);
function updateCreateButton() {
  $("#ws-create").disabled = !$("#ws-parent").value || !$("#ws-name-input").value.trim();
}

$("#ws-open-other").addEventListener("click", async () => {
  const r = await api("/api/workspaces/choose-folder", { method: "POST" });
  if (r.path) switchWorkspace(r.path);
});

$("#ws-create").addEventListener("click", async () => {
  $("#ws-error").textContent = "";
  try {
    const r = await api("/api/workspaces/create", { method: "POST", body: JSON.stringify({ parent: $("#ws-parent").value.trim(), name: $("#ws-name-input").value.trim() }) });
    $("#ws-name-input").value = "";
    $("#ws-parent").value = "";
    onWorkspaceActivated(r.active);
  } catch (e) {
    $("#ws-error").textContent = e.message;
  }
});
$("#ws-name-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#ws-create").click(); });

async function switchWorkspace(path, init = false) {
  $("#ws-error").textContent = "";
  try {
    const r = await api("/api/workspaces/open", { method: "POST", body: JSON.stringify({ path, init }) });
    onWorkspaceActivated(r.active);
  } catch (e) {
    if (e.body?.error === "not_workspace") {
      if (confirm(e.message)) return switchWorkspace(path, true);
      return;
    }
    if ($("#ws-dialog").hidden) alert(e.message); else $("#ws-error").textContent = e.message;
  }
}

/// Everything shown belongs to one workspace: reset it all for the new one.
function onWorkspaceActivated(active) {
  dialogRequired = false;
  $("#ws-dialog").hidden = true;
  $("#ws-btn-name").textContent = active?.name || "No workspace";
  document.title = active?.name ? `localRAG · ${active.name}` : "localRAG";
  closeViewer();
  viewer.doc = null; viewer.docId = null;
  $("#search-results").replaceChildren();
  $("#upload-results").replaceChildren();
  newChat();
  loadModels();
  refreshStatus();
  if ($("#tab-documents").classList.contains("active")) loadDocuments();
  maybePromptEmail();
}

// ---------------------------------------------------------------- uploads

$("#add-docs").addEventListener("click", () => $("#add-docs-input").click());
$("#add-docs-input").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  e.target.value = "";
  const list = $("#upload-results");
  list.replaceChildren();
  const before = (await api("/api/status")).sync?.runs ?? 0;
  const accepted = /\.(pdf|md|markdown|txt)$/i;
  for (const f of files) {
    const row = el("li", { class: "busy" }, `Checking ${f.name}…`);
    list.append(row);
    if (!accepted.test(f.name)) {
      row.className = "err";
      row.textContent = `“${f.name}” isn't a supported file type. localrag indexes PDF (.pdf), Markdown (.md, .markdown) and plain text (.txt).`;
      continue;
    }
    try {
      const r = await api(`/api/documents/upload?name=${encodeURIComponent(f.name)}`, {
        method: "POST", body: f, headers: { "Content-Type": "application/octet-stream" },
      });
      const c = r.check;
      const parts = [`Added ${r.saved_as}${r.renamed ? ` (a file named “${f.name}” already existed)` : ""}`];
      parts.push(c.kind === "pdf" ? `${c.pages} page${c.pages === 1 ? "" : "s"}` : `${c.text_chars.toLocaleString()} characters`);
      if (c.scanned_pages.length) parts.push(`${c.scanned_pages.length} scanned page(s) will need OCR (pages ${c.scanned_pages.join(", ")})`);
      row.className = "ok";
      row.textContent = `${parts.join(" · ")}. Indexing…`;
    } catch (err) {
      row.className = "err";
      row.textContent = err.message;
    }
  }
  if (list.querySelector("li.ok")) await waitForSync(before);
  for (const row of list.querySelectorAll("li.ok")) row.textContent = row.textContent.replace(/ Indexing…$/, " Indexed.");
  loadDocuments();
});

/// Small corpora sync in milliseconds, faster than the status poll notices: wait until a
/// sync that started after `before` (a completed-sync count) has finished.
async function waitForSync(before) {
  for (let i = 0; i < 1200; i++) {
    const { sync } = await api("/api/status");
    if (!sync || (!sync.running && sync.runs > before)) return;
    await new Promise((r) => setTimeout(r, 500));
  }
}

// ---------------------------------------------------------------- OCR

async function runOcr(d) {
  try {
    const before = (await api("/api/status")).sync.runs;
    await api(`/api/documents/${d.id}/ocr`, { method: "POST" });
    $("#upload-results").replaceChildren(el("li", { class: "busy" }, `Running OCR on ${d.path}… progress is shown at the top right.`));
    refreshStatus();
    await waitForSync(before);
    $("#upload-results").replaceChildren(el("li", { class: "ok" }, `OCR finished for ${d.path}.`));
    loadDocuments();
  } catch (e) {
    $("#upload-results").replaceChildren(el("li", { class: "err" }, e.message));
  }
}
$("#ocr-all").addEventListener("click", async () => {
  const before = (await api("/api/status")).sync.runs;
  try { await api("/api/ocr/all", { method: "POST" }); } catch (e) { alert(e.message); return; }
  refreshStatus();
  await waitForSync(before);
  loadDocuments();
});
$("#ocr-cancel").addEventListener("click", () => api("/api/ocr/cancel", { method: "POST" }));

// ---------------------------------------------------------------- Crossref email & settings

async function maybePromptEmail() {
  const s = await api("/api/settings");
  if (s.crossref_email || s.crossref_email_prompted) return;
  $("#email-error").textContent = "";
  $("#email-dialog").hidden = false;
  $("#email-input").focus();
}
$("#email-save").addEventListener("click", async () => {
  try {
    await api("/api/settings", { method: "PUT", body: JSON.stringify({ crossref_email: $("#email-input").value }) });
    $("#email-dialog").hidden = true;
  } catch (e) { $("#email-error").textContent = e.message; }
});
$("#email-input").addEventListener("keydown", (e) => { if (e.key === "Enter") $("#email-save").click(); });
$("#email-skip").addEventListener("click", async () => {
  await api("/api/settings", { method: "PUT", body: JSON.stringify({ crossref_email_prompted: true }) });
  $("#email-dialog").hidden = true;
});

$("#settings-btn").addEventListener("click", openSettings);
$("#settings-close").addEventListener("click", () => ($("#settings-dialog").hidden = true));
async function openSettings() {
  const s = await api("/api/settings");
  $("#settings-error").textContent = "";
  $("#set-email").value = s.crossref_email || "";
  $("#set-pool").textContent = s.crossref_email
    ? "Polite pool: up to 10 DOI lookups and 3 title searches per second. Clear the field to stop sending it."
    : "Public pool: 5 DOI lookups and 1 title search per second. Stored only on this Mac and sent only to Crossref.";
  const ws = s.workspace;
  for (const id of ["#set-model", "#set-numctx", "#set-ocr", "#set-repair", "#set-strict"]) $(id).disabled = !ws;
  if (ws) {
    const { models } = await api("/api/models").catch(() => ({ models: [] }));
    $("#set-model").replaceChildren(...models.map((m) => el("option", { value: m.name }, m.name)));
    $("#set-model").value = ws.chat_model;
    const fill = () => ($("#set-numctx").value = ws.num_ctx[$("#set-model").value] ?? "");
    $("#set-model").onchange = fill;
    fill();
    $("#set-numctx").placeholder = "chosen from this Mac's memory on first use";
    $("#set-ocr").value = ws.ocr;
    $("#set-repair").checked = ws.repair_quotes;
    $("#set-strict").checked = ws.strict;
  }
  $("#settings-dialog").hidden = false;
}
$("#settings-save").addEventListener("click", async () => {
  const body = { crossref_email: $("#set-email").value };
  if (!$("#set-model").disabled) {
    body.chat_model = $("#set-model").value;
    body.ocr = $("#set-ocr").value;
    body.repair_quotes = $("#set-repair").checked;
    body.strict = $("#set-strict").checked;
    const n = parseInt($("#set-numctx").value, 10);
    if (n) body.num_ctx = { [body.chat_model]: n };
  }
  try {
    await api("/api/settings", { method: "PUT", body: JSON.stringify(body) });
    $("#settings-dialog").hidden = true;
    loadModels();
  } catch (e) { $("#settings-error").textContent = e.message; }
});

// ---------------------------------------------------------------- metadata

let metaDoc = null;
$("#meta-close").addEventListener("click", () => ($("#meta-dialog").hidden = true));
async function openMetadata(d) {
  metaDoc = d;
  $("#meta-error").textContent = "";
  $("#meta-file").textContent = d.path;
  renderMetadata(await api(`/api/documents/${d.id}/metadata`));
  $("#meta-dialog").hidden = false;
}
function renderMetadata(m) {
  const csl = m.csl || {};
  const src = { crossref_doi: "matched on Crossref by DOI", crossref_query: "matched on Crossref by title", manual: "edited by you (never overwritten by lookups)", pdf: "from the PDF's own metadata", frontmatter: "from the Markdown front matter" }[m.source];
  $("#meta-state").textContent = m.status === "proposed" ? "Crossref found possible matches: pick the right one, or reject them and fill in the details yourself."
    : m.status === "pending" ? "Not looked up yet: it happens after indexing."
    : m.status === "unmatched" ? "No confident match on Crossref. You can fill in the details below."
    : src ? `Source: ${src}.` : "";
  if (m.error) $("#meta-state").textContent += ` Last lookup failed: ${m.error}`;
  const cands = m.status === "proposed" ? m.candidates || [] : [];
  $("#meta-candidates-section").hidden = !cands.length;
  $("#meta-candidates").replaceChildren(...cands.map((c, i) => {
    const w = c.work;
    const who = (w.author || []).slice(0, 3).map((a) => a.family || a.literal).filter(Boolean).join(", ");
    return el("li", {},
      el("div", { class: "info" }, el("div", {}, w.title || "(untitled)"), el("div", { class: "score" }, `${who}${w.author?.length > 3 ? " et al." : ""} · ${w.year ?? "?"} · ${w["container-title"] || ""} · ${w.DOI} · ${Math.round(c.score * 100)}% match`)),
      el("button", { class: "small primary", onclick: async () => { renderMetadata(await api(`/api/documents/${metaDoc.id}/metadata/resolve`, { method: "POST", body: JSON.stringify({ accept: i }) })); loadDocuments(); } }, "This one"));
  }));
  $("#meta-title-input").value = csl.title || "";
  $("#meta-authors").value = (csl.author || []).map((a) => a.literal || [a.family, a.given].filter(Boolean).join(", ")).join("\n");
  $("#meta-year").value = csl.issued?.["date-parts"]?.[0]?.[0] ?? "";
  $("#meta-pages").value = csl.page || "";
  $("#meta-container").value = csl["container-title"] || "";
  $("#meta-doi").value = csl.DOI || m.doi || "";
}
$("#meta-reject").addEventListener("click", async () => {
  renderMetadata(await api(`/api/documents/${metaDoc.id}/metadata/resolve`, { method: "POST", body: JSON.stringify({ accept: null }) }));
  loadDocuments();
});
$("#meta-refresh").addEventListener("click", async () => {
  $("#meta-error").textContent = "Looking up…";
  try {
    const r = await api(`/api/documents/${metaDoc.id}/metadata/refresh`, { method: "POST" });
    $("#meta-error").textContent = "";
    renderMetadata(r.metadata);
    loadDocuments();
  } catch (e) { $("#meta-error").textContent = e.message; }
});
$("#meta-save").addEventListener("click", async () => {
  const year = parseInt($("#meta-year").value, 10);
  const body = {
    title: $("#meta-title-input").value.trim() || null,
    authors: $("#meta-authors").value.split("\n").map((a) => a.trim()).filter(Boolean),
    year: Number.isFinite(year) ? year : null,
    page: $("#meta-pages").value.trim() || null,
    container: $("#meta-container").value.trim() || null,
    doi: $("#meta-doi").value.trim() || null,
  };
  try {
    renderMetadata(await api(`/api/documents/${metaDoc.id}`.concat("/metadata"), { method: "PATCH", body: JSON.stringify(body) }));
    loadDocuments();
  } catch (e) { $("#meta-error").textContent = e.message; }
});

// ---------------------------------------------------------------- start

(async function start() {
  const { active } = await api("/api/workspaces");
  if (!active) {
    $("#status").innerHTML = `<span class="dot bad"></span>No workspace open`;
    showWorkspaceDialog();
    return;
  }
  onWorkspaceActivated(active);
  loadSessions();
})();
