//! Display labels for citations: `[Author Year, p. N]` (ARCHITECTURE.md §10).

use serde_json::Value;

/// "Smith 2021", "Smith & Jones 2021", "Smith et al. 2021", falling back to the file name.
/// Phase 4 replaces this with Crossref metadata and `a`/`b` suffixes for collisions.
pub fn doc_label(cite_label: Option<&str>, csl_json: Option<&str>, path: &str) -> String {
    if let Some(l) = cite_label.filter(|l| !l.is_empty()) {
        return l.to_string();
    }
    if let Some(l) = csl_json.and_then(|c| serde_json::from_str::<Value>(c).ok()).and_then(|v| from_csl(&v)) {
        return l;
    }
    from_filename(path)
}

fn from_csl(v: &Value) -> Option<String> {
    let names: Vec<String> = v["author"]
        .as_array()?
        .iter()
        .filter_map(|a| a["family"].as_str().or(a["literal"].as_str()).map(surname))
        .collect();
    let year = v["issued"]["date-parts"][0][0].as_i64()?;
    let who = match names.as_slice() {
        [] => return None,
        [a] => a.clone(),
        [a, b] => format!("{a} & {b}"),
        [a, ..] => format!("{a} et al."),
    };
    Some(format!("{who} {year}"))
}

/// Last word of a literal name ("A. Tester" → "Tester").
fn surname(name: &str) -> String {
    name.split_whitespace().last().unwrap_or(name).trim_matches(',').to_string()
}

/// Zotero-style names ("Parry et al. - 2011 - Title.pdf") become "Parry et al. 2011";
/// anything else becomes the file stem, shortened.
fn from_filename(path: &str) -> String {
    let file = path.rsplit('/').next().unwrap_or(path);
    let stem = file.rsplit_once('.').map_or(file, |(s, _)| s);
    let parts: Vec<&str> = stem.split(" - ").collect();
    if parts.len() >= 2 {
        let year = parts[1].trim();
        if year.len() == 4 && year.chars().all(|c| c.is_ascii_digit()) {
            return format!("{} {year}", parts[0].trim());
        }
    }
    let short: String = stem.chars().take(40).collect();
    if short.len() < stem.len() { format!("{short}…") } else { short }
}

/// "p. 12", "pp. 12–13", "ll. 4–9" or "§Methods".
pub fn locator(labels: (Option<&str>, Option<&str>), pdf_pages: (Option<i64>, Option<i64>), lines: (Option<i64>, Option<i64>), section: Option<&str>) -> String {
    let start = labels.0.map(str::to_string).or(pdf_pages.0.map(|p| p.to_string()));
    let end = labels.1.map(str::to_string).or(pdf_pages.1.map(|p| p.to_string()));
    match (start, end) {
        (Some(a), Some(b)) if a != b => format!("pp. {a}–{b}"),
        (Some(a), _) => format!("p. {a}"),
        _ => match (section, lines) {
            (Some(s), _) if !s.is_empty() => format!("§{}", s.rsplit(" > ").next().unwrap_or(s)),
            (_, (Some(a), Some(b))) if a != b => format!("ll. {a}–{b}"),
            (_, (Some(a), _)) => format!("l. {a}"),
            _ => String::new(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn labels() {
        assert_eq!(doc_label(None, None, "Parry et al. - 2011 - Cancer survivors.pdf"), "Parry et al. 2011");
        assert_eq!(doc_label(None, None, "notes/attention.md"), "attention");
        let csl = r#"{"author":[{"literal":"A. Tester"},{"literal":"B. Writer"}],"issued":{"date-parts":[[2026]]}}"#;
        assert_eq!(doc_label(None, Some(csl), "x.md"), "Tester & Writer 2026");
        assert_eq!(doc_label(Some("Smith 2021a"), Some(csl), "x.md"), "Smith 2021a");
    }

    #[test]
    fn locators() {
        assert_eq!(locator((Some("12"), Some("13")), (Some(3), Some(4)), (None, None), None), "pp. 12–13");
        assert_eq!(locator((None, None), (Some(3), Some(3)), (None, None), None), "p. 3");
        assert_eq!(locator((None, None), (None, None), (Some(4), Some(9)), Some("Intro > Methods")), "§Methods");
        assert_eq!(locator((None, None), (None, None), (Some(4), Some(9)), None), "ll. 4–9");
    }
}
