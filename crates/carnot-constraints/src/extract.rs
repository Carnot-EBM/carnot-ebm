//! Pluggable constraint extraction from text — Rust port of Python extractors.
//!
//! ## For Researchers
//!
//! Provides a trait-based extractor architecture where each domain (arithmetic,
//! logic) has its own extractor struct. Each extractor parses specific content
//! types and returns `ConstraintResult` values that can carry metadata about
//! whether the extracted constraint is satisfied.
//!
//! ## For Engineers
//!
//! This module ports the Python `carnot.pipeline.extract` module to Rust for
//! 10x performance (NFR-01). The architecture mirrors Python's Protocol-based
//! design using Rust traits:
//!
//! - `ConstraintResult`: A single extracted constraint with type, description,
//!   satisfaction status, and domain-specific metadata.
//! - `ConstraintExtractor`: Trait defining the `extract()` interface.
//! - `ArithmeticExtractor`: Parses "X + Y = Z" and "X - Y = Z" patterns.
//! - `LogicExtractor`: Parses "If P then Q", "X but not Y", "X or Y",
//!   "X cannot Y", and "All X are Y" patterns.
//! - `AutoExtractor`: Combines all extractors and deduplicates results.
//!
//! Note: `CodeExtractor` and `NLExtractor` are NOT ported because they require
//! Python AST parsing and NLP capabilities respectively. They remain in the
//! Python pipeline. This Rust port focuses on the extractors that are pure
//! regex/text matching and benefit most from Rust's performance.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002

use regex::Regex;
use std::collections::HashMap;

/// A single extracted constraint ready for verification.
///
/// ## For Engineers
///
/// Each constraint extracted from text is wrapped in this struct. The
/// `metadata` field holds domain-specific details (e.g., for arithmetic:
/// operands, claimed result, correct result, whether satisfied). The
/// `verified` field is a convenience flag set by extractors that can
/// verify inline during extraction (like ArithmeticExtractor which
/// computes the correct answer immediately).
///
/// Spec: REQ-VERIFY-001
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    /// Category tag, e.g. "arithmetic", "implication", "exclusion".
    pub constraint_type: String,
    /// Human-readable summary of what the constraint checks.
    pub description: String,
    /// Whether this constraint is satisfied (if determinable at extraction time).
    /// `None` means satisfaction cannot be determined without external verification.
    pub verified: Option<bool>,
    /// Domain-specific details as string key-value pairs.
    /// For arithmetic: "a", "b", "operator", "claimed_result", "correct_result", "satisfied".
    /// For logic: "antecedent", "consequent", "raw", etc.
    pub metadata: HashMap<String, String>,
}

impl ConstraintResult {
    /// Create a new ConstraintResult with the given type and description.
    pub fn new(constraint_type: &str, description: &str) -> Self {
        Self {
            constraint_type: constraint_type.to_string(),
            description: description.to_string(),
            verified: None,
            metadata: HashMap::new(),
        }
    }

    /// Builder method: set the verified flag.
    pub fn with_verified(mut self, verified: bool) -> Self {
        self.verified = Some(verified);
        self
    }

    /// Builder method: insert a metadata key-value pair.
    pub fn with_meta(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Trait for domain-specific constraint extractors.
///
/// ## For Engineers
///
/// Any struct implementing this trait can be used as a pluggable extractor
/// in the verification pipeline. The trait mirrors Python's `ConstraintExtractor`
/// Protocol.
///
/// Spec: REQ-VERIFY-001
pub trait ConstraintExtractor: Send + Sync {
    /// List of domain tags this extractor handles (e.g., ["arithmetic"]).
    fn supported_domains(&self) -> &[&str];

    /// Extract constraints from `text`.
    ///
    /// If `domain` is `Some`, the extractor should skip processing if the
    /// domain is not in `supported_domains()`. If `None`, extract all
    /// applicable constraints.
    fn extract(&self, text: &str, domain: Option<&str>) -> Vec<ConstraintResult>;
}

// ---------------------------------------------------------------------------
// ArithmeticExtractor — port of Python ArithmeticExtractor
// ---------------------------------------------------------------------------

/// Extract and verify "X + Y = Z" and "X - Y = Z" arithmetic claims from text.
///
/// ## For Engineers
///
/// Scans text for patterns like "47 + 28 = 75" using regex. Supports addition
/// and subtraction with positive and negative integers. Each match is verified
/// by direct computation and returned as a `ConstraintResult` with satisfaction
/// status in the `verified` field and metadata.
///
/// This is a direct port of Python's `ArithmeticExtractor` from
/// `carnot.pipeline.extract`, using identical regex patterns and verification
/// logic to ensure cross-language result equivalence.
///
/// Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
pub struct ArithmeticExtractor {
    /// Compiled regex for matching arithmetic expressions.
    /// Pattern: (-?\d+)\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)
    pattern: Regex,
}

impl ArithmeticExtractor {
    /// Create a new ArithmeticExtractor with precompiled regex.
    pub fn new() -> Self {
        Self {
            pattern: Regex::new(r"(-?\d+)\s*([+\-])\s*(-?\d+)\s*=\s*(-?\d+)").unwrap(),
        }
    }
}

impl Default for ArithmeticExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintExtractor for ArithmeticExtractor {
    fn supported_domains(&self) -> &[&str] {
        &["arithmetic"]
    }

    fn extract(&self, text: &str, domain: Option<&str>) -> Vec<ConstraintResult> {
        if let Some(d) = domain {
            if !self.supported_domains().contains(&d) {
                return Vec::new();
            }
        }

        let mut results = Vec::new();

        for cap in self.pattern.captures_iter(text) {
            let a: i64 = cap[1].parse().unwrap();
            let op = &cap[2];
            let b_raw: i64 = cap[3].parse().unwrap();
            let claimed: i64 = cap[4].parse().unwrap();

            // Apply the operator to get the effective operand (mirrors Python logic).
            let b = if op == "+" { b_raw } else { -b_raw };
            let correct = a + b;
            let satisfied = claimed == correct;

            let description = if satisfied {
                format!("{a} {op} {b_raw} = {claimed}")
            } else {
                format!("{a} {op} {b_raw} = {claimed} (correct: {correct})")
            };

            let result = ConstraintResult::new("arithmetic", &description)
                .with_verified(satisfied)
                .with_meta("a", &a.to_string())
                .with_meta("b", &b.to_string())
                .with_meta("operator", op)
                .with_meta("claimed_result", &claimed.to_string())
                .with_meta("correct_result", &correct.to_string())
                .with_meta("satisfied", &satisfied.to_string());

            results.push(result);
        }

        results
    }
}

// ---------------------------------------------------------------------------
// LogicExtractor — port of Python LogicExtractor
// ---------------------------------------------------------------------------

/// Extract logical relationship claims from text.
///
/// ## For Engineers
///
/// Scans text for conditional patterns ("if ... then ...", "if ..., ..."),
/// mutual exclusion patterns ("X but not Y"), disjunction ("either X or Y"),
/// negation ("X cannot Y"), and universal quantifiers ("All X are Y").
///
/// Returns ConstraintResult objects tagged with the appropriate constraint type.
/// These constraints are structural — they identify logical relationships but
/// do not verify truth values (that requires external knowledge).
///
/// Direct port of Python's `LogicExtractor` with identical regex patterns.
///
/// Spec: REQ-VERIFY-001, SCENARIO-VERIFY-002
pub struct LogicExtractor {
    /// Precompiled regex patterns for each logical pattern type.
    implication_re: Regex,
    exclusion_re: Regex,
    disjunction_re: Regex,
    negation_re: Regex,
    universal_re: Regex,
}

impl LogicExtractor {
    /// Create a new LogicExtractor with precompiled regex patterns.
    pub fn new() -> Self {
        Self {
            // "if X then Y" or "if X, Y" or "if X, then Y"
            implication_re: Regex::new(
                r"(?i)^if\s+(.+?)(?:,\s*(?:then\s+)?|\s+then\s+)(.+?)\.?$",
            )
            .unwrap(),
            // "X but not Y"
            exclusion_re: Regex::new(r"(?i)^(.+?)\s+but\s+not\s+(.+?)\.?$").unwrap(),
            // "either X or Y" or "X or Y"
            disjunction_re: Regex::new(r"(?i)^(?:either\s+)?(.+?)\s+or\s+(.+?)\.?$").unwrap(),
            // "X cannot/can't/does not Y"
            negation_re: Regex::new(
                r"(?i)^(.+?)\s+(?:cannot|can't|can not|do not|does not|don't|doesn't)\s+(.+?)\.?$",
            )
            .unwrap(),
            // "All X are/is Y"
            universal_re: Regex::new(r"(?i)^all\s+(.+?)\s+(?:are|is)\s+(.+?)\.?$").unwrap(),
        }
    }

    /// Normalize text: lowercase and collapse whitespace (mirrors Python _normalize).
    fn normalize(text: &str) -> String {
        let re = Regex::new(r"\s+").unwrap();
        re.replace_all(text.trim(), " ").to_lowercase()
    }

    /// Split text into sentences on sentence-ending punctuation.
    /// Uses a simple approach instead of lookbehind (not supported by Rust regex).
    fn split_sentences(text: &str) -> Vec<String> {
        let re = Regex::new(r"[.!?]\s+").unwrap();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Vec::new();
        }
        // Split on punctuation+whitespace, but keep the sentence text.
        // We need to reconstruct sentences by finding split points.
        let mut sentences = Vec::new();
        let mut last = 0;
        for m in re.find_iter(trimmed) {
            // Include the punctuation character in the sentence.
            let end = m.start() + 1; // include the [.!?]
            let sentence = trimmed[last..end].trim().to_string();
            if !sentence.is_empty() {
                sentences.push(sentence);
            }
            last = m.end();
        }
        // Remaining text after last split.
        let remaining = trimmed[last..].trim().to_string();
        if !remaining.is_empty() {
            sentences.push(remaining);
        }
        sentences
    }

    fn extract_implication(&self, sentence: &str) -> Vec<ConstraintResult> {
        let s = Self::normalize(sentence);
        if let Some(cap) = self.implication_re.captures(&s) {
            let ante = cap[1].trim().to_string();
            let cons = cap[2].trim().to_string();
            return vec![ConstraintResult::new(
                "implication",
                &format!("If {ante}, then {cons}"),
            )
            .with_meta("antecedent", &ante)
            .with_meta("consequent", &cons)
            .with_meta("raw", sentence.trim())];
        }
        Vec::new()
    }

    fn extract_exclusion(&self, sentence: &str) -> Vec<ConstraintResult> {
        let s = Self::normalize(sentence);
        if let Some(cap) = self.exclusion_re.captures(&s) {
            let positive = cap[1].trim().to_string();
            let negative = cap[2].trim().to_string();
            return vec![
                ConstraintResult::new("exclusion", &format!("{positive} but not {negative}"))
                    .with_meta("positive", &positive)
                    .with_meta("negative", &negative)
                    .with_meta("raw", sentence.trim()),
            ];
        }
        Vec::new()
    }

    fn extract_disjunction(&self, sentence: &str) -> Vec<ConstraintResult> {
        let s = Self::normalize(sentence);
        if let Some(cap) = self.disjunction_re.captures(&s) {
            let left = cap[1].trim().to_string();
            let right = cap[2].trim().to_string();
            return vec![
                ConstraintResult::new("disjunction", &format!("{left} or {right}"))
                    .with_meta("left", &left)
                    .with_meta("right", &right)
                    .with_meta("raw", sentence.trim()),
            ];
        }
        Vec::new()
    }

    fn extract_negation(&self, sentence: &str) -> Vec<ConstraintResult> {
        let s = Self::normalize(sentence);
        if let Some(cap) = self.negation_re.captures(&s) {
            let subject = cap[1].trim().to_string();
            let predicate = cap[2].trim().to_string();
            return vec![
                ConstraintResult::new("negation", &format!("{subject} cannot {predicate}"))
                    .with_meta("subject", &subject)
                    .with_meta("predicate", &predicate)
                    .with_meta("raw", sentence.trim()),
            ];
        }
        Vec::new()
    }

    fn extract_universal(&self, sentence: &str) -> Vec<ConstraintResult> {
        let s = Self::normalize(sentence);
        if let Some(cap) = self.universal_re.captures(&s) {
            let category = cap[1].trim().to_string();
            let property = cap[2].trim().to_string();
            return vec![ConstraintResult::new(
                "universal",
                &format!("All {category} are {property}"),
            )
            .with_meta("category", &category)
            .with_meta("property", &property)
            .with_meta("raw", sentence.trim())];
        }
        Vec::new()
    }
}

impl Default for LogicExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintExtractor for LogicExtractor {
    fn supported_domains(&self) -> &[&str] {
        &["logic"]
    }

    fn extract(&self, text: &str, domain: Option<&str>) -> Vec<ConstraintResult> {
        if let Some(d) = domain {
            if !self.supported_domains().contains(&d) {
                return Vec::new();
            }
        }

        let mut results = Vec::new();
        for sent in Self::split_sentences(text) {
            // Try each pattern in order, matching Python's extraction order.
            // Only the first matching pattern fires per sentence to avoid
            // double-counting (e.g., "If A then B" could match both
            // implication and disjunction via "A then B or ...").
            let extracted = self.extract_implication(&sent);
            if !extracted.is_empty() {
                results.extend(extracted);
                continue;
            }
            let extracted = self.extract_exclusion(&sent);
            if !extracted.is_empty() {
                results.extend(extracted);
                continue;
            }
            let extracted = self.extract_negation(&sent);
            if !extracted.is_empty() {
                results.extend(extracted);
                continue;
            }
            let extracted = self.extract_universal(&sent);
            if !extracted.is_empty() {
                results.extend(extracted);
                continue;
            }
            // Disjunction is last because it's the most greedy pattern
            // ("X or Y" matches many things).
            let extracted = self.extract_disjunction(&sent);
            if !extracted.is_empty() {
                results.extend(extracted);
            }
        }
        results
    }
}

// ---------------------------------------------------------------------------
// AutoExtractor — combines all extractors
// ---------------------------------------------------------------------------

/// Combines all domain extractors and auto-detects applicable domains.
///
/// ## For Engineers
///
/// Holds a registry of `ConstraintExtractor` trait objects. When `extract()`
/// is called without a domain hint, it runs ALL extractors and merges their
/// results (deduplicating by description). When a domain is specified, only
/// extractors that support that domain are invoked.
///
/// Spec: REQ-VERIFY-001, REQ-VERIFY-002, SCENARIO-VERIFY-002
pub struct AutoExtractor {
    extractors: Vec<Box<dyn ConstraintExtractor>>,
}

impl AutoExtractor {
    /// Create an AutoExtractor with the default set of extractors:
    /// ArithmeticExtractor and LogicExtractor.
    pub fn new() -> Self {
        Self {
            extractors: vec![
                Box::new(ArithmeticExtractor::new()),
                Box::new(LogicExtractor::new()),
            ],
        }
    }

    /// Register an additional domain extractor.
    pub fn add_extractor(&mut self, extractor: Box<dyn ConstraintExtractor>) {
        self.extractors.push(extractor);
    }

    /// Union of all registered extractors' supported domains.
    pub fn supported_domains(&self) -> Vec<&str> {
        let mut domains = Vec::new();
        for ext in &self.extractors {
            for &d in ext.supported_domains() {
                if !domains.contains(&d) {
                    domains.push(d);
                }
            }
        }
        domains
    }
}

impl Default for AutoExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl ConstraintExtractor for AutoExtractor {
    fn supported_domains(&self) -> &[&str] {
        // AutoExtractor supports all domains, but since we need a static
        // slice reference, we return a sentinel. Callers should use the
        // inherent `supported_domains()` method instead.
        &[]
    }

    fn extract(&self, text: &str, domain: Option<&str>) -> Vec<ConstraintResult> {
        let mut results = Vec::new();
        let mut seen_descriptions = std::collections::HashSet::new();

        for ext in &self.extractors {
            if let Some(d) = domain {
                if !ext.supported_domains().contains(&d) {
                    continue;
                }
            }
            for result in ext.extract(text, domain) {
                if seen_descriptions.insert(result.description.clone()) {
                    results.push(result);
                }
            }
        }
        results
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ArithmeticExtractor tests (mirrors Python TestArithmeticExtractor) ---

    #[test]
    fn test_correct_addition() {
        // REQ-VERIFY-001: Correct arithmetic claim is marked satisfied.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("47 + 28 = 75", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].constraint_type, "arithmetic");
        assert_eq!(results[0].verified, Some(true));
        assert_eq!(results[0].metadata["correct_result"], "75");
    }

    #[test]
    fn test_wrong_addition() {
        // REQ-VERIFY-001: Incorrect arithmetic claim is marked unsatisfied.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("47 + 28 = 76", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verified, Some(false));
        assert_eq!(results[0].metadata["correct_result"], "75");
    }

    #[test]
    fn test_subtraction() {
        // REQ-VERIFY-001: Subtraction claims are correctly verified.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("15 - 7 = 8", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verified, Some(true));
    }

    #[test]
    fn test_wrong_subtraction() {
        // REQ-VERIFY-001: Wrong subtraction is detected.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("15 - 7 = 9", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verified, Some(false));
    }

    #[test]
    fn test_negative_operand() {
        // REQ-VERIFY-001: Negative operands are handled.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("-3 + 10 = 7", None);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].verified, Some(true));
    }

    #[test]
    fn test_multiple_claims() {
        // REQ-VERIFY-001: Multiple arithmetic claims in one text.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("First: 2 + 3 = 5. Then: 10 - 4 = 6.", None);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.verified == Some(true)));
    }

    #[test]
    fn test_no_arithmetic() {
        // REQ-VERIFY-001: Text with no arithmetic returns empty.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("The sky is blue.", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_empty_text() {
        // REQ-VERIFY-001: Empty string returns empty.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_domain_filter_skip() {
        // SCENARIO-VERIFY-002: Domain filter skips non-matching domains.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("2 + 3 = 5", Some("code"));
        assert!(results.is_empty());
    }

    #[test]
    fn test_domain_filter_match() {
        // SCENARIO-VERIFY-002: Domain filter passes matching domain.
        let ext = ArithmeticExtractor::new();
        let results = ext.extract("2 + 3 = 5", Some("arithmetic"));
        assert_eq!(results.len(), 1);
    }

    // --- LogicExtractor tests (mirrors Python TestLogicExtractor) ---

    #[test]
    fn test_if_then() {
        // REQ-VERIFY-001: 'If P then Q' implication is extracted.
        let ext = LogicExtractor::new();
        let results = ext.extract("If it rains, then the ground is wet.", None);
        let implications: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "implication")
            .collect();
        assert_eq!(implications.len(), 1);
        assert_eq!(implications[0].metadata["antecedent"], "it rains");
        assert_eq!(implications[0].metadata["consequent"], "the ground is wet");
    }

    #[test]
    fn test_if_comma() {
        // REQ-VERIFY-001: 'If P, Q' pattern (comma separator) is extracted.
        let ext = LogicExtractor::new();
        let results = ext.extract("If it rains, the ground is wet.", None);
        let implications: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "implication")
            .collect();
        assert_eq!(implications.len(), 1);
    }

    #[test]
    fn test_exclusion() {
        // REQ-VERIFY-001: 'X but not Y' exclusion is extracted.
        let ext = LogicExtractor::new();
        let results = ext.extract("Mammals but not reptiles.", None);
        let exclusions: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "exclusion")
            .collect();
        assert_eq!(exclusions.len(), 1);
        assert_eq!(exclusions[0].metadata["positive"], "mammals");
        assert_eq!(exclusions[0].metadata["negative"], "reptiles");
    }

    #[test]
    fn test_disjunction() {
        // REQ-VERIFY-001: 'X or Y' disjunction is extracted.
        let ext = LogicExtractor::new();
        let results = ext.extract("Either cats or dogs.", None);
        let disj: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "disjunction")
            .collect();
        assert_eq!(disj.len(), 1);
    }

    #[test]
    fn test_negation() {
        // REQ-VERIFY-001: 'X cannot Y' negation is extracted.
        let ext = LogicExtractor::new();
        let results = ext.extract("Penguins cannot fly.", None);
        let negs: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "negation")
            .collect();
        assert_eq!(negs.len(), 1);
        assert_eq!(negs[0].metadata["subject"], "penguins");
        assert_eq!(negs[0].metadata["predicate"], "fly");
    }

    #[test]
    fn test_universal() {
        // REQ-VERIFY-001: 'All X are Y' universal quantifier is extracted.
        let ext = LogicExtractor::new();
        let results = ext.extract("All mammals are warm-blooded.", None);
        let univs: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "universal")
            .collect();
        assert_eq!(univs.len(), 1);
        assert_eq!(univs[0].metadata["category"], "mammals");
        assert_eq!(univs[0].metadata["property"], "warm-blooded");
    }

    #[test]
    fn test_no_logic() {
        // REQ-VERIFY-001: Text with no logical patterns returns empty.
        // "The sky is blue." has "is" which could match disjunction loosely,
        // but since it's a single sentence without "or"/"but not"/etc., it
        // should return empty for logic-specific patterns.
        let ext = LogicExtractor::new();
        let results = ext.extract("Hello world!", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_logic_empty_text() {
        // REQ-VERIFY-001: Empty string returns empty.
        let ext = LogicExtractor::new();
        let results = ext.extract("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_multiple_implications() {
        // REQ-VERIFY-001: Multiple logical claims in one text.
        let ext = LogicExtractor::new();
        let results = ext.extract("If A then B. If B then C.", None);
        let implications: Vec<_> = results
            .iter()
            .filter(|r| r.constraint_type == "implication")
            .collect();
        assert_eq!(implications.len(), 2);
    }

    #[test]
    fn test_logic_domain_filter() {
        // SCENARIO-VERIFY-002: Domain filter skips non-matching domains.
        let ext = LogicExtractor::new();
        let results = ext.extract("If A then B.", Some("code"));
        assert!(results.is_empty());
    }

    // --- AutoExtractor tests ---

    #[test]
    fn test_auto_combines_arithmetic_and_logic() {
        // REQ-VERIFY-002: AutoExtractor combines results from multiple extractors.
        let ext = AutoExtractor::new();
        let results = ext.extract("2 + 3 = 5. If it rains, then the ground is wet.", None);
        let types: std::collections::HashSet<_> =
            results.iter().map(|r| r.constraint_type.as_str()).collect();
        assert!(types.contains("arithmetic"));
        assert!(types.contains("implication"));
    }

    #[test]
    fn test_auto_domain_filter() {
        // SCENARIO-VERIFY-002: Domain filter restricts to specific extractor.
        let ext = AutoExtractor::new();
        let results = ext.extract(
            "2 + 3 = 5. If it rains, then the ground is wet.",
            Some("arithmetic"),
        );
        assert!(results.iter().all(|r| r.constraint_type == "arithmetic"));
    }

    #[test]
    fn test_auto_empty_text() {
        // REQ-VERIFY-001: Empty text returns empty from all extractors.
        let ext = AutoExtractor::new();
        let results = ext.extract("", None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_auto_deduplication() {
        // REQ-VERIFY-002: Same description from multiple extractors is deduplicated.
        let ext = AutoExtractor::new();
        let results = ext.extract("All mammals are warm-blooded.", None);
        let descriptions: Vec<_> = results.iter().map(|r| &r.description).collect();
        let unique: std::collections::HashSet<_> = descriptions.iter().collect();
        assert_eq!(descriptions.len(), unique.len());
    }

    #[test]
    fn test_auto_supported_domains() {
        // REQ-VERIFY-001: supported_domains includes both arithmetic and logic.
        let ext = AutoExtractor::new();
        let domains = ext.supported_domains();
        assert!(domains.contains(&"arithmetic"));
        assert!(domains.contains(&"logic"));
    }
}
