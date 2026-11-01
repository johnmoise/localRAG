//! Brain files (docs/ARCHITECTURE.md §12): `brain/<model>.md`, one per reasoning model.
//! Every line is a heading, a `[THOUGHT]`, a `[FACT]` or blank; anything else is invalid
//! (reported and ignored). `[FACT]` lines are written only by the program.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::scan::sha256_str;
use crate::workspace::Workspace;

/// One citation of a fact: where the quote is, by content hash (§13).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FactCitation {
    /// "Vaswani 2017, p. 6"
    pub label: String,
    pub quote: String,
    pub doc_sha256: String,
    pub chunk_text_sha: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FactLine {
    pub id: String,
    pub claim: String,
    pub citations: Vec<FactCitation>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Line {
    /// `#`, `##` or `###`, with its text.
    Heading { level: usize, text: String },
    Thought { text: String },
    Fact { fact: FactLine, raw: String },
    Blank,
    Invalid { raw: String, error: String },
}

impl Line {
    pub fn raw(&self) -> String {
        match self {
            Line::Heading { level, text } => format!("{} {text}", "#".repeat(*level)),
            Line::Thought { text } => format!("[THOUGHT] {text}"),
            Line::Fact { raw, .. } => raw.clone(),
            Line::Blank => String::new(),
            Line::Invalid { raw, .. } => raw.clone(),
        }
    }
}

pub fn parse_line(raw: &str) -> Line {
    let t = raw.trim_end();
    if t.trim().is_empty() {
        return Line::Blank;
    }
    if let Some(rest) = t.strip_prefix("[THOUGHT]") {
        return Line::Thought { text: rest.trim().to_string() };
    }
    if t.starts_with("[FACT]") {
        return match parse_fact(t) {
            Ok(fact) => Line::Fact { fact, raw: t.to_string() },
            Err(e) => Line::Invalid { raw: t.to_string(), error: format!("{e:#}") },
        };
    }
    let hashes = t.chars().take_while(|c| *c == '#').count();
    if (1..=3).contains(&hashes) && t[hashes..].starts_with(' ') && !t[hashes..].trim().is_empty() {
        return Line::Heading { level: hashes, text: t[hashes..].trim().to_string() };
    }
    Line::Invalid { raw: t.to_string(), error: "not a heading, [THOUGHT], [FACT] or blank line".into() }
}

pub fn parse(text: &str) -> Vec<Line> {
    text.lines().map(parse_line).collect()
}

pub fn render(lines: &[Line]) -> String {
    let mut s: String = lines.iter().map(Line::raw).collect::<Vec<_>>().join("\n");
    s.push('\n');
    s
}

/// `|` separates fields, so it (and the escape character) are escaped inside them.
fn esc(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").replace('\\', "\\\\").replace('|', "\\|")
}

fn split_fields(s: &str) -> Vec<String> {
    let mut out = vec![String::new()];
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => {
                if let Some(n) = chars.next() {
                    out.last_mut().unwrap().push(n);
                }
            }
            '|' => out.push(String::new()),
            _ => out.last_mut().unwrap().push(c),
        }
    }
    out.into_iter().map(|f| f.trim().to_string()).collect()
}

/// `[FACT] F-0012 | claim | Label, p. 6 "quote" <doc:sha256, chunk:sha256> | …`
pub fn render_fact(f: &FactLine) -> String {
    let mut s = format!("[FACT] {} | {}", f.id, esc(&f.claim));
    for c in &f.citations {
        s.push_str(&format!(" | {} \"{}\" <doc:{}, chunk:{}>", esc(&c.label.replace('"', "'")), esc(&c.quote), c.doc_sha256, c.chunk_text_sha));
    }
    s
}

fn parse_fact(line: &str) -> anyhow::Result<FactLine> {
    let fields = split_fields(line.trim_start_matches("[FACT]"));
    if fields.len() < 3 {
        bail!("a [FACT] line needs an ID, a claim and at least one citation, separated by |");
    }
    let id = fields[0].clone();
    if !is_fact_id(&id) {
        bail!("“{id}” isn't a fact ID (F-0001)");
    }
    let claim = fields[1].clone();
    if claim.is_empty() {
        bail!("empty claim");
    }
    let citations = fields[2..].iter().map(|c| parse_citation(c)).collect::<anyhow::Result<Vec<_>>>()?;
    Ok(FactLine { id, claim, citations })
}

fn parse_citation(c: &str) -> anyhow::Result<FactCitation> {
    let bad = || anyhow::anyhow!("citation “{c}” should look like: Label, p. 6 \"quote\" <doc:…, chunk:…>");
    let open = c.rfind(" <doc:").ok_or_else(bad)?;
    let refs = c[open + 6..].strip_suffix('>').ok_or_else(bad)?;
    let (doc, chunk) = refs.split_once(", chunk:").ok_or_else(bad)?;
    let head = c[..open].trim_end();
    let q_end = head.strip_suffix('"').ok_or_else(bad)?;
    let q_start = q_end.find('"').ok_or_else(bad)?;
    let hex = |s: &str| s.len() == 64 && s.chars().all(|c| c.is_ascii_hexdigit());
    if !hex(doc) || !hex(chunk) {
        bail!("citation hashes must be 64 hex characters");
    }
    Ok(FactCitation {
        label: q_end[..q_start].trim().to_string(),
        quote: q_end[q_start + 1..].to_string(),
        doc_sha256: doc.to_string(),
        chunk_text_sha: chunk.to_string(),
    })
}

pub fn is_fact_id(s: &str) -> bool {
    s.strip_prefix("F-").is_some_and(|n| n.len() >= 4 && n.chars().all(|c| c.is_ascii_digit()))
}

/// "F-0012" → 12
pub fn id_number(id: &str) -> Option<u32> {
    id.strip_prefix("F-").and_then(|n| n.parse().ok())
}

/// `deepseek-r1:8b` → `deepseek-r1_8b`
pub fn file_stem(model: &str) -> String {
    model.chars().map(|c| if c.is_ascii_alphanumeric() || "._-".contains(c) { c } else { '_' }).collect()
}

pub fn brain_dir(ws: &Workspace) -> PathBuf {
    ws.root.join("brain")
}

pub fn path_for(ws: &Workspace, model: &str) -> PathBuf {
    brain_dir(ws).join(format!("{}.md", file_stem(model)))
}

pub fn history_dir(ws: &Workspace, model: &str) -> PathBuf {
    ws.state_dir().join("brain-history").join(file_stem(model))
}

pub fn header(model: &str) -> String {
    format!("# Brain · {model}")
}

/// The model named in a brain file's header line.
pub fn model_in_header(text: &str) -> Option<String> {
    text.lines().next()?.strip_prefix("# Brain · ").map(|m| m.trim().to_string())
}

/// A brain as read from disk.
#[derive(Debug, Clone)]
pub struct Brain {
    pub model: String,
    pub path: PathBuf,
    pub text: String,
    pub lines: Vec<Line>,
    /// sha256 of `text`: writes must name the version they were based on.
    pub hash: String,
    pub exists: bool,
}

pub fn load(ws: &Workspace, model: &str) -> anyhow::Result<Brain> {
    let path = path_for(ws, model);
    let (text, exists) = match std::fs::read_to_string(&path) {
        Ok(t) => (t, true),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => (format!("{}\n\n## Unsorted\n", header(model)), false),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    Ok(Brain { model: model.to_string(), lines: parse(&text), hash: sha256_str(&text), path, text, exists })
}

/// Models with a brain file in this workspace (from each file's header).
pub fn models_with_brains(ws: &Workspace) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(brain_dir(ws))
        .into_iter()
        .flatten()
        .flatten()
        .filter(|e| e.path().extension().is_some_and(|x| x == "md"))
        .filter_map(|e| std::fs::read_to_string(e.path()).ok().and_then(|t| model_in_header(&t)))
        .collect();
    out.sort();
    out
}

/// One lock per brain file: every write in this process goes through it.
fn lock_for(path: &std::path::Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<PathBuf, Arc<Mutex<()>>>>> = OnceLock::new();
    LOCKS.get_or_init(Default::default).lock().expect("brain locks").entry(path.to_path_buf()).or_default().clone()
}

#[derive(Debug)]
pub struct Conflict;

impl std::fmt::Display for Conflict {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "the brain file changed since it was read (edited elsewhere?); reload and try again")
    }
}

impl std::error::Error for Conflict {}

/// Reads the brain, applies `change` to its lines, and writes the result, all under the
/// file's lock. With `expected_hash`, the write is refused ([`Conflict`]) if the file isn't
/// the version the caller saw. `history`: keep the previous version for Undo.
pub fn update<T>(ws: &Workspace, model: &str, expected_hash: Option<&str>, history: bool, change: impl FnOnce(&mut Vec<Line>) -> anyhow::Result<T>) -> anyhow::Result<(T, Brain)> {
    let path = path_for(ws, model);
    let lock = lock_for(&path);
    let _guard = lock.lock().expect("brain lock");
    let before = load(ws, model)?;
    if expected_hash.is_some_and(|h| h != before.hash) {
        return Err(Conflict.into());
    }
    let mut lines = before.lines.clone();
    let out = change(&mut lines)?;
    let text = render(&lines);
    if text != before.text || !before.exists {
        if history && before.exists {
            save_history(ws, model, &before.text)?;
        }
        write_atomic(&path, &text)?;
    }
    Ok((out, load(ws, model)?))
}

fn write_atomic(path: &std::path::Path, text: &str) -> anyhow::Result<()> {
    let dir = path.parent().context("brain path")?;
    std::fs::create_dir_all(dir)?;
    ensure_gitignore(dir)?;
    let tmp = dir.join(format!(".{}.tmp{}", path.file_name().unwrap_or_default().to_string_lossy(), std::process::id()));
    std::fs::write(&tmp, text)?;
    std::fs::rename(&tmp, path).with_context(|| format!("writing {}", path.display()))?;
    Ok(())
}

/// Brain files quote the corpus, so like `corpus/` they're kept out of git unless you
/// delete or edit this file (§14).
fn ensure_gitignore(dir: &std::path::Path) -> anyhow::Result<()> {
    let gi = dir.join(".gitignore");
    if !gi.exists() {
        std::fs::write(gi, "# Brain files quote your documents; delete this file to version them.\n*\n")?;
    }
    Ok(())
}

fn save_history(ws: &Workspace, model: &str, text: &str) -> anyhow::Result<()> {
    let dir = history_dir(ws, model);
    std::fs::create_dir_all(&dir)?;
    let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    std::fs::write(dir.join(format!("{stamp:015}.md")), text)?;
    Ok(())
}

/// Previous versions, newest first.
pub fn history(ws: &Workspace, model: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(history_dir(ws, model)).into_iter().flatten().flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "md")).collect();
    v.sort();
    v.reverse();
    v
}

/// Restores the most recent saved version and drops it from the history. Returns it.
pub fn undo(ws: &Workspace, model: &str) -> anyhow::Result<Brain> {
    let path = path_for(ws, model);
    let lock = lock_for(&path);
    let _guard = lock.lock().expect("brain lock");
    let Some(prev) = history(ws, model).into_iter().next() else { bail!("nothing to undo") };
    let text = std::fs::read_to_string(&prev)?;
    write_atomic(&path, &text)?;
    std::fs::remove_file(prev)?;
    load(ws, model)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cit() -> FactCitation {
        FactCitation { label: "Vaswani 2017, p. 6".into(), quote: "complexity per layer O(n²·d) | per \"layer\"".into(), doc_sha256: "a".repeat(64), chunk_text_sha: "b".repeat(64) }
    }

    #[test]
    fn fact_lines_round_trip_with_escapes() {
        let f = FactLine { id: "F-0012".into(), claim: "Self-attention is O(n²) | quadratic \\ costly.".into(), citations: vec![cit(), cit()] };
        let raw = render_fact(&f);
        assert!(raw.starts_with("[FACT] F-0012 | Self-attention is O(n²) \\| quadratic \\\\ costly. | Vaswani 2017, p. 6 \""), "{raw}");
        match parse_line(&raw) {
            Line::Fact { fact, .. } => assert_eq!(fact, f),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn line_types() {
        let lines = parse("# Brain · m\n\n## Cost\n[THOUGHT] F-0001 and F-0002 differ.\nrandom text\n#### too deep\n[FACT] F-1 | x | y\n");
        assert!(matches!(lines[0], Line::Heading { level: 1, .. }));
        assert!(matches!(lines[1], Line::Blank));
        assert!(matches!(lines[3], Line::Thought { .. }));
        assert!(matches!(lines[4], Line::Invalid { .. }));
        assert!(matches!(lines[5], Line::Invalid { .. }));
        assert!(matches!(lines[6], Line::Invalid { .. }), "bad ID and citation");
        assert_eq!(model_in_header("# Brain · deepseek-r1:8b\n"), Some("deepseek-r1:8b".into()));
        assert_eq!(file_stem("deepseek-r1:8b"), "deepseek-r1_8b");
    }
}
