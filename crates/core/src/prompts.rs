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

pub const SUGGEST_REFS_SYSTEM: &str = "A researcher's document collection does not answer their question. You are given works that documents in the collection cite but that are NOT in the collection, numbered R1, R2, and so on, most similar to the question first, some with an excerpt of their abstract. Choose up to 3 of them that would most likely help answer the question, each with a one-sentence reason based only on the citation and abstract shown. Choose only from the numbered list; never name any other work. If none of them would help, choose none. Also give 2 to 4 topics or search keywords worth looking for (not paper titles).";

pub fn suggest_refs_request(question: &str, candidates: &[crate::references::MissingReference]) -> String {
    let mut s = format!("Question: {question}\n\nCited by the collection but not in it:\n");
    for (i, c) in candidates.iter().enumerate() {
        let what = match (&c.title, &c.container) {
            (Some(t), Some(j)) => format!("{t}. {j}"),
            (Some(t), None) => t.clone(),
            _ => truncate(&c.raw, 220),
        };
        s.push_str(&format!("R{} {}: {what} (cited by {} document(s) in the collection)\n", i + 1, c.label, c.cited_in.len()));
        if let Some(a) = &c.abstract_text {
            s.push_str(&format!("   Abstract: {}\n", truncate(a, 400)));
        }
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

// ---------------------------------------------------------------- second brain (§12)

pub const PROBE_SYSTEM: &str = "Answer the question from your own knowledge only, in one to three sentences, with specific facts and numbers where you know them. If you don't know, say \"I don't know\".";

pub const KNOWN_SYSTEM: &str = "You compare a claim with an answer someone gave without looking anything up. Reply same=true only if the answer states the same specific information as the claim (the same finding, and the same numbers where the claim has numbers). A vaguer, different, hedged or \"I don't know\" answer is same=false.";

pub fn known_request(claim: &str, unaided: &str) -> String {
    format!("Claim: {claim}\n\nAnswer given without sources: {unaided}")
}

pub const CONFLICT_SYSTEM: &str = "For each numbered statement, say whether it contradicts the new statement: both cannot be true at the same time about the same thing. Statements about different populations, methods or conditions do not contradict. Items are numbered from 0.";

pub fn conflict_request(claim: &str, others: &[(String, String)]) -> String {
    let mut s = format!("New statement: {claim}\n\nExisting statements:\n");
    for (i, (_, c)) in others.iter().enumerate() {
        s.push_str(&format!("{i}. {c}\n"));
    }
    s
}

pub const ORGANIZE_SYSTEM: &str = "You organize your own research notes (your \"brain\"). It holds verified facts, written as [FACT F-0012] lines, under Markdown headings, with [THOUGHT] lines: your notes on how facts relate (agreements, apparent contradictions and why, open questions).

Rearrange it so related facts sit together:
- Group facts under clear # / ## / ### headings; reorder facts within and between sections.
- Add, revise or remove [THOUGHT] lines. Thoughts are interpretation: mention facts by ID (F-0012) and never restate a fact as if it were verified.
- Keep the first line (# Brain · <model>) as it is.
- Write each fact ONLY as [FACT <id>], exactly once each, every fact included. Never write a fact's text: the program fills it in.
- Every line must be a heading, a [THOUGHT] line, a [FACT <id>] line or empty.

Return JSON: {\"lines\": [ ...one string per line... ]}.";

pub fn organize_request(layout: &str) -> String {
    format!("Your brain now:\n\n{layout}")
}

pub const AGENT_SYSTEM: &str = "You are a careful research assistant working over a researcher's own document collection. You have tools:
- read_brain(query): the verified facts you have learned before. Call it first.
- ask_corpus(question): asks the collection one focused question and returns a cited answer. Each usable citation has an ID like [C3] with its source and exact quote.
- propose_fact(claim, citations, reason): proposes a durable fact for your brain, supported by citation IDs from ask_corpus results. The researcher approves or rejects it later.

How to work:
1. read_brain with the question.
2. Break what the brain doesn't cover into a few focused sub-questions and ask_corpus each (at most a handful).
3. When an answer establishes something general and reusable for later questions (not trivia about one sentence), propose_fact with a one-sentence claim that the cited quotes directly support, and say why it is worth keeping.
4. Finish with an answer to the original question, without calling tools. Support every claim with citations written as [C3] (from ask_corpus) or [F12] (from your brain). Never invent citations. If the collection can't answer, say so plainly.";

pub const REPORT_SYSTEM: &str = "You are a careful research assistant writing a short report that answers ONE research question for a researcher, grounded in their own document collection.

Your brain has just been reorganized and read into this conversation: Markdown headings, [THOUGHT] lines (your earlier interpretations) and [FACT] lines. Facts are verified against the collection and are your most trusted evidence.

Plan before you write:
1. Decide what a good answer to THIS question must cover. Then pick only the facts and thoughts that bear on it. Most of the brain may be irrelevant: leave out every topic, fact and thought that does not help answer the question, however interesting.
2. Build the answer on the relevant facts first. Cite a fact directly as [F12] (or [F-0012]) wherever it supports a point; do not re-ask the collection for something a fact already establishes.
3. Only for what the relevant facts leave open, call ask_corpus(question) with focused sub-questions, one at a time. Each answer comes with citation IDs like [C3]. Stop asking once you can answer the question well.

Then, without calling tools, write the report:
- A short title line, then 4 to 5 paragraphs of connected exposition that build one argument answering the research question: open with the direct answer, develop it with evidence, address disagreements or limits, and close by restating what the evidence supports. Each paragraph should follow from the previous one; it is not a list of topics.
- Every factual statement needs a citation: [F#] for brain facts, [C#] for ask_corpus citations. Never cite a [THOUGHT], and never invent citations. Use thoughts only to guide how you connect the evidence.
- Where the evidence doesn't settle something, say so plainly instead of filling the gap.

Your only tool is ask_corpus; you cannot add facts to the brain.";

pub fn report_tools() -> serde_json::Value {
    serde_json::json!([
        { "type": "function", "function": { "name": "ask_corpus", "description": "Ask the document collection one focused question; returns a cited answer with citation IDs like C3.",
            "parameters": { "type": "object", "properties": { "question": { "type": "string" } }, "required": ["question"] } } }
    ])
}

pub fn agent_tools() -> serde_json::Value {
    serde_json::json!([
        { "type": "function", "function": { "name": "read_brain", "description": "Read your verified facts relevant to a query.",
            "parameters": { "type": "object", "properties": { "query": { "type": "string" } }, "required": ["query"] } } },
        { "type": "function", "function": { "name": "ask_corpus", "description": "Ask the document collection one focused question; returns a cited answer with citation IDs like C3.",
            "parameters": { "type": "object", "properties": { "question": { "type": "string" } }, "required": ["question"] } } },
        { "type": "function", "function": { "name": "propose_fact", "description": "Propose a durable, general fact for your brain, supported by citation IDs from ask_corpus.",
            "parameters": { "type": "object", "properties": {
                "claim": { "type": "string", "description": "One sentence, directly supported by the cited quotes." },
                "citations": { "type": "array", "items": { "type": "string" }, "description": "Citation IDs such as C3." },
                "reason": { "type": "string", "description": "Why this is worth remembering for later questions." } },
              "required": ["claim", "citations", "reason"] } } }
    ])
}
