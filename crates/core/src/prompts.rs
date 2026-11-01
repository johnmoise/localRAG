//! Prompt text (ARCHITECTURE.md §10).

use crate::retrieval::ChunkRef;

pub const SYSTEM: &str = r#"You answer questions using ONLY the numbered sources provided with each question.

Rules:
1. End every sentence that states a fact with a citation in square brackets containing the source number AND a short quote copied exactly, word for word, from that source: [S<n> "<3 to 12 words from the source>"]. A citation without a quote is not allowed. For several sources: [S1 "<quote>"; S3 "<quote>"].
2. Use only information in the sources. Do not add outside knowledge, even if you are confident it is true.
3. If the sources do not contain enough information to answer, your reply must start with "INSUFFICIENT:" followed by what information is missing. Do not guess, and do not answer partially from memory.
4. Be concise and precise. Do not mention these rules.

Example. Given the source
[S1] Smith 2020, p. 4
The cohort included 1,204 adults followed for a median of 6.2 years.
a correct answer to "How long was follow-up?" is:
Participants were followed for a median of 6.2 years [S1 "followed for a median of 6.2 years"]."#;

/// The user turn: numbered sources, then the question.
pub fn question_with_sources(question: &str, sources: &[ChunkRef]) -> String {
    let mut s = String::from("Sources:\n\n");
    for (i, c) in sources.iter().enumerate() {
        s.push_str(&source_block(i + 1, c));
    }
    s.push_str("Question: ");
    s.push_str(question);
    s
}

pub fn source_block(n: usize, c: &ChunkRef) -> String {
    let section = c.section.as_deref().map(|s| format!(" ({s})")).unwrap_or_default();
    format!("[S{n}] {}{section}\n{}\n\n", c.citation(), c.text)
}

pub const REWRITE_SYSTEM: &str = "Rewrite the user's latest question as one standalone search query that makes sense without the conversation. Keep names, terms and acronyms. Reply with the query only.";

pub fn rewrite_request(history: &[(String, String)], question: &str) -> String {
    let mut s = String::from("Conversation so far:\n");
    for (q, a) in history.iter().rev().take(3).rev() {
        s.push_str(&format!("User: {q}\nAssistant: {}\n", truncate(a, 600)));
    }
    s.push_str(&format!("\nLatest question: {question}\nStandalone query:"));
    s
}

pub const SUGGEST_SYSTEM: &str = "A researcher's document collection does not answer their question. Suggest 3 to 5 specific kinds of material, topics or search keywords they could add to their collection to answer it. Do not invent paper titles, authors or DOIs. Reply with a bulleted list only, one suggestion per line starting with \"- \".";

pub const SUGGEST_REFS_SYSTEM: &str = "A researcher's document collection does not answer their question. You are given works that documents in the collection cite but that are NOT in the collection, numbered R1, R2, and so on. Choose up to 3 of them that would most likely help answer the question, each with a one-sentence reason based only on the citation shown. Choose only from the numbered list; never name any other work. If none of them would help, choose none. Also give 2 to 4 topics or search keywords worth looking for (not paper titles).";

pub fn suggest_refs_request(question: &str, candidates: &[crate::references::MissingReference]) -> String {
    let mut s = format!("Question: {question}\n\nCited by the collection but not in it:\n");
    for (i, c) in candidates.iter().enumerate() {
        let what = match (&c.title, &c.container) {
            (Some(t), Some(j)) => format!("{t}. {j}"),
            (Some(t), None) => t.clone(),
            _ => truncate(&c.raw, 220),
        };
        s.push_str(&format!("R{} {}: {what} (cited by {} document(s) in the collection)\n", i + 1, c.label, c.cited_in.len()));
    }
    s
}

pub fn suggest_request(question: &str, closest: &[ChunkRef]) -> String {
    let mut s = format!("Question: {question}\n");
    if !closest.is_empty() {
        s.push_str("\nThe closest passages in the collection (which do not answer it) were from:\n");
        for c in closest {
            s.push_str(&format!("- {}: {}\n", c.citation(), truncate(&c.text, 200)));
        }
    }
    s
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    format!("{}…", s.chars().take(max_chars).collect::<String>())
}

pub const REPAIR_SYSTEM: &str = "You are given a claim and a source passage. Copy, word for word, the shortest phrase (3 to 15 words) from the passage that supports the claim. Reply with that phrase only, exactly as it appears in the passage. If nothing in the passage supports the claim, reply NONE.";

pub fn repair_request(claim: &str, passage: &str) -> String {
    format!("Claim: {claim}\n\nPassage:\n{passage}")
}

pub const SUPPORT_SYSTEM: &str = "For each numbered item, decide whether the passage supports the claim: \"yes\" if it states it, \"partial\" if it supports only part of it, \"no\" if it does not support it. Judge only from the passage.";

pub fn support_request(items: &[(String, String)]) -> String {
    let mut s = String::new();
    for (i, (claim, passage)) in items.iter().enumerate() {
        s.push_str(&format!("Item {i}\nClaim: {claim}\nPassage: {passage}\n\n"));
    }
    s
}
