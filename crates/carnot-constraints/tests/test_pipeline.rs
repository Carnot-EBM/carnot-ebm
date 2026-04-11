//! Integration tests for the Rust verification pipeline.
//!
//! These tests use the same 20 inputs as the Python pipeline tests to verify
//! cross-language result equivalence for the arithmetic and logic extractors.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-002, SCENARIO-VERIFY-004

use carnot_constraints::extract::{
    ArithmeticExtractor, AutoExtractor, ConstraintExtractor, LogicExtractor,
};
use carnot_constraints::pipeline::VerifyPipeline;

// ---------------------------------------------------------------------------
// Cross-language equivalence: Arithmetic (mirrors Python TestArithmeticExtractor)
// ---------------------------------------------------------------------------

/// Test inputs 1-10: Arithmetic extraction, matching Python test_pipeline_extract.py
#[test]
fn cross_lang_01_correct_addition() {
    // REQ-VERIFY-001: "47 + 28 = 75" -> satisfied=true, correct_result=75
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("47 + 28 = 75", None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].constraint_type, "arithmetic");
    assert_eq!(results[0].verified, Some(true));
    assert_eq!(results[0].metadata["correct_result"], "75");
}

#[test]
fn cross_lang_02_wrong_addition() {
    // REQ-VERIFY-001: "47 + 28 = 76" -> satisfied=false, correct_result=75
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("47 + 28 = 76", None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified, Some(false));
    assert_eq!(results[0].metadata["correct_result"], "75");
}

#[test]
fn cross_lang_03_subtraction() {
    // REQ-VERIFY-001: "15 - 7 = 8" -> satisfied=true
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("15 - 7 = 8", None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified, Some(true));
}

#[test]
fn cross_lang_04_wrong_subtraction() {
    // REQ-VERIFY-001: "15 - 7 = 9" -> satisfied=false
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("15 - 7 = 9", None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified, Some(false));
}

#[test]
fn cross_lang_05_negative_operand() {
    // REQ-VERIFY-001: "-3 + 10 = 7" -> satisfied=true
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("-3 + 10 = 7", None);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].verified, Some(true));
}

#[test]
fn cross_lang_06_multiple_arithmetic() {
    // REQ-VERIFY-001: Two correct claims in one text
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("First: 2 + 3 = 5. Then: 10 - 4 = 6.", None);
    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|r| r.verified == Some(true)));
}

#[test]
fn cross_lang_07_no_arithmetic() {
    // REQ-VERIFY-001: No arithmetic in text -> empty
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("The sky is blue.", None);
    assert!(results.is_empty());
}

#[test]
fn cross_lang_08_empty_text_arithmetic() {
    // REQ-VERIFY-001: Empty string -> empty
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("", None);
    assert!(results.is_empty());
}

#[test]
fn cross_lang_09_domain_filter_skip() {
    // SCENARIO-VERIFY-002: Domain "code" skips arithmetic extractor
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("2 + 3 = 5", Some("code"));
    assert!(results.is_empty());
}

#[test]
fn cross_lang_10_domain_filter_match() {
    // SCENARIO-VERIFY-002: Domain "arithmetic" matches
    let ext = ArithmeticExtractor::new();
    let results = ext.extract("2 + 3 = 5", Some("arithmetic"));
    assert_eq!(results.len(), 1);
}

// ---------------------------------------------------------------------------
// Cross-language equivalence: Logic (mirrors Python TestLogicExtractor)
// ---------------------------------------------------------------------------

/// Test inputs 11-20: Logic extraction, matching Python test_pipeline_extract.py
#[test]
fn cross_lang_11_if_then() {
    // REQ-VERIFY-001: "If P then Q" implication
    let ext = LogicExtractor::new();
    let results = ext.extract("If it rains, then the ground is wet.", None);
    let imps: Vec<_> = results
        .iter()
        .filter(|r| r.constraint_type == "implication")
        .collect();
    assert_eq!(imps.len(), 1);
    assert_eq!(imps[0].metadata["antecedent"], "it rains");
    assert_eq!(imps[0].metadata["consequent"], "the ground is wet");
}

#[test]
fn cross_lang_12_if_comma() {
    // REQ-VERIFY-001: "If P, Q" comma-separated
    let ext = LogicExtractor::new();
    let results = ext.extract("If it rains, the ground is wet.", None);
    let imps: Vec<_> = results
        .iter()
        .filter(|r| r.constraint_type == "implication")
        .collect();
    assert_eq!(imps.len(), 1);
}

#[test]
fn cross_lang_13_exclusion() {
    // REQ-VERIFY-001: "X but not Y"
    let ext = LogicExtractor::new();
    let results = ext.extract("Mammals but not reptiles.", None);
    let excl: Vec<_> = results
        .iter()
        .filter(|r| r.constraint_type == "exclusion")
        .collect();
    assert_eq!(excl.len(), 1);
    assert_eq!(excl[0].metadata["positive"], "mammals");
    assert_eq!(excl[0].metadata["negative"], "reptiles");
}

#[test]
fn cross_lang_14_disjunction() {
    // REQ-VERIFY-001: "Either X or Y"
    let ext = LogicExtractor::new();
    let results = ext.extract("Either cats or dogs.", None);
    let disj: Vec<_> = results
        .iter()
        .filter(|r| r.constraint_type == "disjunction")
        .collect();
    assert_eq!(disj.len(), 1);
}

#[test]
fn cross_lang_15_negation() {
    // REQ-VERIFY-001: "X cannot Y"
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
fn cross_lang_16_universal() {
    // REQ-VERIFY-001: "All X are Y"
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
fn cross_lang_17_no_logic() {
    // REQ-VERIFY-001: No logical patterns -> empty
    let ext = LogicExtractor::new();
    let results = ext.extract("Hello world!", None);
    assert!(results.is_empty());
}

#[test]
fn cross_lang_18_empty_text_logic() {
    // REQ-VERIFY-001: Empty string -> empty
    let ext = LogicExtractor::new();
    let results = ext.extract("", None);
    assert!(results.is_empty());
}

#[test]
fn cross_lang_19_multiple_implications() {
    // REQ-VERIFY-001: Multiple "If then" in one text
    let ext = LogicExtractor::new();
    let results = ext.extract("If A then B. If B then C.", None);
    let imps: Vec<_> = results
        .iter()
        .filter(|r| r.constraint_type == "implication")
        .collect();
    assert_eq!(imps.len(), 2);
}

#[test]
fn cross_lang_20_logic_domain_filter() {
    // SCENARIO-VERIFY-002: Domain "code" skips logic extractor
    let ext = LogicExtractor::new();
    let results = ext.extract("If A then B.", Some("code"));
    assert!(results.is_empty());
}

// ---------------------------------------------------------------------------
// Pipeline integration tests
// ---------------------------------------------------------------------------

#[test]
fn pipeline_correct_arithmetic_verified() {
    // REQ-VERIFY-003: Correct arithmetic passes the full pipeline.
    let pipeline = VerifyPipeline::default();
    let result = pipeline.verify("What is 47 + 28?", "The answer is 47 + 28 = 75.");
    assert!(result.verified);
    assert!(result.violations.is_empty());
    assert!(result.energy < 1e-10);
    assert!(result.certificate.is_verified());
    assert_eq!(result.certificate.num_constraints, 1);
    assert_eq!(result.certificate.num_satisfied, 1);
    assert_eq!(result.certificate.num_violated, 0);
}

#[test]
fn pipeline_wrong_arithmetic_violated() {
    // REQ-VERIFY-003: Wrong arithmetic fails the full pipeline.
    let pipeline = VerifyPipeline::default();
    let result = pipeline.verify("What is 47 + 28?", "The answer is 47 + 28 = 76.");
    assert!(!result.verified);
    assert_eq!(result.violations.len(), 1);
    assert!(result.energy > 0.0);
    assert!(!result.certificate.is_verified());
}

#[test]
fn pipeline_mixed_constraints() {
    // REQ-VERIFY-003: Mixed correct and incorrect constraints.
    let pipeline = VerifyPipeline::default();
    let result = pipeline.verify(
        "Compute",
        "2 + 3 = 5. 10 - 4 = 7. If it rains, then the ground is wet.",
    );
    // 2+3=5 correct, 10-4=7 wrong (should be 6), logic is structural.
    assert!(!result.verified);
    assert_eq!(result.violations.len(), 1);
    // Total constraints: 1 arithmetic correct + 1 arithmetic wrong + 1 implication
    assert!(result.constraints.len() >= 3);
}

#[test]
fn pipeline_no_constraints_vacuous() {
    // REQ-VERIFY-003: No extractable constraints -> vacuously verified.
    let pipeline = VerifyPipeline::default();
    let result = pipeline.verify("Tell me a joke", "Why did the chicken cross the road?");
    assert!(result.verified);
    assert!(result.constraints.is_empty());
    assert!(result.certificate.is_verified());
}

#[test]
fn pipeline_certificate_json_roundtrip() {
    // REQ-VERIFY-003: Certificate survives JSON serialization.
    let pipeline = VerifyPipeline::default();
    let result = pipeline.verify("Test", "2 + 3 = 5. 10 - 4 = 7.");
    let json = result.certificate.to_json().unwrap();
    let restored = carnot_constraints::verify::VerificationCertificate::from_json(&json).unwrap();
    assert_eq!(result.certificate.status, restored.status);
    assert_eq!(result.certificate.num_constraints, restored.num_constraints);
    assert_eq!(result.certificate.num_violated, restored.num_violated);
}

#[test]
fn pipeline_custom_extractor() {
    // REQ-VERIFY-002: Pipeline works with custom extractor.
    let pipeline = VerifyPipeline::with_extractor(Box::new(ArithmeticExtractor::new()));
    let result = pipeline.verify("Mixed", "2 + 3 = 5. If A then B.");
    // Only arithmetic extractor, so logic patterns are not extracted.
    assert!(result
        .constraints
        .iter()
        .all(|c| c.constraint_type == "arithmetic"));
    assert_eq!(result.constraints.len(), 1);
}

#[test]
fn pipeline_auto_extractor_union() {
    // REQ-VERIFY-002: AutoExtractor covers both arithmetic and logic.
    let ext = AutoExtractor::new();
    let domains = ext.supported_domains();
    assert!(domains.contains(&"arithmetic"));
    assert!(domains.contains(&"logic"));
}
