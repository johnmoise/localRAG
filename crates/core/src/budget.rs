//! Token counting and the context budget (ARCHITECTURE.md §11).

use rusqlite::Connection;
use serde::Serialize;
use tokenizers::Tokenizer;

use crate::db;
use crate::llm::ModelInfo;

/// Extra tokens per chat message for role markers and the chat template.
pub const PER_MESSAGE_OVERHEAD: usize = 8;
/// Output reserve for plain and thinking models (thinking tokens count against num_ctx).
pub const OUTPUT_RESERVE: usize = 1024;
pub const OUTPUT_RESERVE_THINKING: usize = 4096;
/// Safety margin applied to estimated (not exact) counts.
const ESTIMATE_MARGIN: f32 = 1.10;
const DEFAULT_CHARS_PER_TOKEN: f32 = 3.5;

pub enum TokenCounter {
    /// Hugging Face tokenizer for the model family: exact (verified in Phase 0).
    Exact(Box<Tokenizer>),
    /// Characters per token, calibrated from Ollama's `prompt_eval_count`.
    Estimate { chars_per_token: f32 },
}

impl TokenCounter {
    /// Uses `third_party/tokenizers/<family>.json` (or the bundled copy) when present.
    pub fn for_model(conn: &Connection, info: &ModelInfo) -> Self {
        let file = (!info.family.is_empty()).then(|| crate::resources::find(&format!("tokenizers/{}.json", info.family))).flatten();
        if let Some(t) = file.and_then(|f| Tokenizer::from_file(f).ok()) {
            return TokenCounter::Exact(Box::new(t));
        }
        let chars_per_token = db::get_meta(conn, &calibration_key(&info.name))
            .ok()
            .flatten()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_CHARS_PER_TOKEN);
        TokenCounter::Estimate { chars_per_token }
    }

    pub fn count(&self, text: &str) -> usize {
        match self {
            TokenCounter::Exact(t) => t.encode(text, false).map(|e| e.len()).unwrap_or_else(|_| estimate(text, DEFAULT_CHARS_PER_TOKEN)),
            TokenCounter::Estimate { chars_per_token } => estimate(text, *chars_per_token),
        }
    }

    pub fn is_exact(&self) -> bool {
        matches!(self, TokenCounter::Exact(_))
    }
}

fn estimate(text: &str, chars_per_token: f32) -> usize {
    ((text.chars().count() as f32 / chars_per_token) * ESTIMATE_MARGIN).ceil() as usize
}

fn calibration_key(model: &str) -> String {
    format!("chars_per_token:{model}")
}

/// Records an observed chars-per-token ratio for `model` (moving average), from a prompt of
/// `chars` characters that Ollama counted as `tokens`.
pub fn calibrate(conn: &Connection, model: &str, chars: usize, tokens: u64) -> anyhow::Result<()> {
    if tokens < 64 {
        return Ok(());
    }
    let observed = chars as f32 / tokens as f32;
    let key = calibration_key(model);
    let previous: Option<f32> = db::get_meta(conn, &key)?.and_then(|v| v.parse().ok());
    let next = previous.map_or(observed, |p| 0.7 * p + 0.3 * observed);
    db::set_meta(conn, &key, &format!("{next:.3}"))
}

/// Default `num_ctx` from unified memory, capped at the model's maximum (§11).
pub fn default_num_ctx(ram_bytes: u64, model_max: Option<u32>) -> u32 {
    let gb = ram_bytes / (1024 * 1024 * 1024);
    let by_ram = match gb {
        0..=11 => 8_192,
        12..=23 => 16_384,
        _ => 32_768,
    };
    model_max.map_or(by_ram, |m| by_ram.min(m))
}

/// The workspace's `num_ctx` for `model`. On first use, picks the RAM-based default, saves
/// it to the workspace config, and returns `true` as the second value.
pub fn resolve_num_ctx(ws: &mut crate::workspace::Workspace, llm: &dyn crate::llm::Llm, model: &str) -> anyhow::Result<(u32, bool)> {
    if let Some(n) = ws.config.num_ctx.get(model) {
        return Ok((*n, false));
    }
    let info = llm.model_info(model)?;
    let n = default_num_ctx(system_ram(), info.context_length);
    ws.config.num_ctx.insert(model.to_string(), n);
    ws.save_config()?;
    Ok((n, true))
}

const FALLBACK_RAM: u64 = 16 * 1024 * 1024 * 1024;

/// Physical memory in bytes (macOS `hw.memsize`).
#[cfg(target_os = "macos")]
pub fn system_ram() -> u64 {
    let mut value: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    let rc = unsafe {
        libc::sysctlbyname(name.as_ptr(), (&mut value as *mut u64).cast(), &mut len, std::ptr::null_mut(), 0)
    };
    if rc == 0 { value } else { FALLBACK_RAM }
}

/// Physical memory in bytes (`GlobalMemoryStatusEx`).
#[cfg(windows)]
pub fn system_ram() -> u64 {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut status = MEMORYSTATUSEX { dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32, ..Default::default() };
    match unsafe { GlobalMemoryStatusEx(&mut status) } {
        Ok(()) => status.ullTotalPhys,
        Err(_) => FALLBACK_RAM,
    }
}

#[cfg(not(any(target_os = "macos", windows)))]
pub fn system_ram() -> u64 {
    FALLBACK_RAM
}

/// Submit blocked: the question plus required grounding can't fit (§11).
#[derive(Debug, Clone, Serialize)]
pub struct OverBudget {
    pub over_by: usize,
    pub question_tokens: usize,
    pub question_limit: usize,
    pub num_ctx: u32,
}

impl std::fmt::Display for OverBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Over token count by {} tokens: the question is {} tokens; the limit for questions with this model is {} (num_ctx {})",
            self.over_by, self.question_tokens, self.question_limit, self.num_ctx
        )
    }
}

impl std::error::Error for OverBudget {}

/// What a request must always fit: everything except history and the optional sources.
#[derive(Debug, Clone, Copy)]
pub struct FixedCosts {
    pub num_ctx: u32,
    pub system: usize,
    pub output_reserve: usize,
    pub min_grounding: usize,
    pub question: usize,
}

impl FixedCosts {
    pub fn total(&self) -> usize {
        self.system + self.output_reserve + self.min_grounding + self.question + 2 * PER_MESSAGE_OVERHEAD
    }

    pub fn check(&self) -> Result<(), OverBudget> {
        let total = self.total();
        let limit = self.num_ctx as usize;
        if total <= limit {
            return Ok(());
        }
        Err(OverBudget {
            over_by: total - limit,
            question_tokens: self.question,
            question_limit: limit.saturating_sub(total - self.question),
            num_ctx: self.num_ctx,
        })
    }

    /// Tokens left for history and sources once the fixed costs (minus the grounding
    /// minimum, which sources then use) are paid.
    pub fn available_for_context(&self) -> usize {
        (self.num_ctx as usize).saturating_sub(self.total() - self.min_grounding)
    }
}

/// Something left out of the context (§11). Always shown to the user.
#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ContextWarning {
    HistoryDropped { turns: usize, tokens: usize },
    SourcesDropped { dropped: usize, relevant: usize },
    NeighborsDropped { passages: usize },
    /// Ollama evaluated fewer prompt tokens than were sent.
    Truncated { sent: usize, evaluated: u64 },
}

impl std::fmt::Display for ContextWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ContextWarning::HistoryDropped { turns, tokens } => write!(
                f, "Context full: {turns} earliest turn{} of this chat ({tokens} tokens) {} left out.",
                if *turns == 1 { "" } else { "s" }, if *turns == 1 { "was" } else { "were" }),
            ContextWarning::SourcesDropped { dropped, relevant } => write!(
                f, "{dropped} of {relevant} relevant passages didn't fit; the answer used the top {}.", relevant - dropped),
            ContextWarning::NeighborsDropped { passages } => write!(f, "Surrounding context for {passages} passages was left out."),
            ContextWarning::Truncated { sent, evaluated } => write!(
                f, "The model received less than was sent ({evaluated} of ~{sent} tokens); the answer may be missing information."),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn over_budget_reports_by_how_much() {
        let fixed = FixedCosts { num_ctx: 4096, system: 300, output_reserve: 1024, min_grounding: 1500, question: 2000 };
        let err = fixed.check().unwrap_err();
        assert_eq!(err.over_by, fixed.total() - 4096);
        assert_eq!(err.question_limit, 2000 - err.over_by);
        assert!(err.to_string().starts_with(&format!("Over token count by {} tokens", err.over_by)));
        let ok = FixedCosts { question: 100, ..fixed };
        assert!(ok.check().is_ok());
        assert_eq!(ok.available_for_context(), 4096 - (300 + 1024 + 100 + 16));
    }

    #[test]
    fn num_ctx_defaults_follow_ram() {
        let gb = 1024 * 1024 * 1024;
        assert_eq!(default_num_ctx(8 * gb, None), 8_192);
        assert_eq!(default_num_ctx(16 * gb, Some(262_144)), 16_384);
        assert_eq!(default_num_ctx(64 * gb, None), 32_768);
        assert_eq!(default_num_ctx(64 * gb, Some(4_096)), 4_096);
    }

    #[test]
    fn estimates_carry_a_margin() {
        let c = TokenCounter::Estimate { chars_per_token: 4.0 };
        assert_eq!(c.count(&"x".repeat(400)), 110);
    }
}
