//! Rust verification pipeline — constraint extraction + composed energy verification.
//!
//! ## For Researchers
//!
//! Wires constraint extraction and energy-based verification into a single
//! `VerifyPipeline` struct. Given a question and response text, the pipeline:
//! 1. Extracts constraints from the response using pluggable extractors.
//! 2. Evaluates constraints with satisfied/violated classification.
//! 3. Computes total energy from any violated constraints.
//! 4. Returns a `PipelineResult` with full decomposition and a
//!    `VerificationCertificate`.
//!
//! ## For Engineers
//!
//! This is the Rust port of Python's `VerifyRepairPipeline.verify()` path.
//! Repair is NOT ported — it requires LLM inference which stays in Python.
//! This module provides the 10x-faster verification path (NFR-01) that Python
//! can call via PyO3 for the hot inner loop.
//!
//! Usage:
//! ```rust,ignore
//! use carnot_constraints::pipeline::VerifyPipeline;
//!
//! let pipeline = VerifyPipeline::default();
//! let result = pipeline.verify(
//!     "What is 47 + 28?",
//!     "The answer is 47 + 28 = 75.",
//! );
//! assert!(result.verified);
//! assert_eq!(result.violations.len(), 0);
//! ```
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, SCENARIO-VERIFY-004

use crate::extract::{AutoExtractor, ConstraintExtractor, ConstraintResult};
use crate::verify::{CertificateStatus, VerificationCertificate};

/// Result of verifying a single response against extracted constraints.
///
/// ## For Engineers
///
/// After the pipeline extracts constraints from a response and evaluates
/// each one, this struct captures the outcome. The `verified` flag is true
/// only when every extracted constraint with a determinable satisfaction
/// status is satisfied. Constraints with `verified = None` (indeterminate)
/// are included in the `constraints` list but do not affect the `verified`
/// flag.
///
/// The `energy` field is a simple sum of penalty values for violated
/// constraints (1.0 per violation). For energy-backed constraints that use
/// the full `ComposedEnergy` pathway, use the `VerificationCertificate`
/// in `certificate`.
///
/// Spec: REQ-VERIFY-003, SCENARIO-VERIFY-004
#[derive(Debug, Clone)]
pub struct PipelineResult {
    /// True if all deterministically-checkable constraints are satisfied.
    pub verified: bool,
    /// All extracted constraints with their evaluation results.
    pub constraints: Vec<ConstraintResult>,
    /// Total energy score: sum of 1.0 per violation. 0.0 = all satisfied.
    pub energy: f64,
    /// Subset of constraints that failed verification.
    pub violations: Vec<ConstraintResult>,
    /// Serializable verification certificate with full decomposition.
    pub certificate: VerificationCertificate,
}

/// The main verification pipeline: extraction + evaluation.
///
/// ## For Engineers
///
/// Holds a list of `ConstraintExtractor` trait objects. When `verify()` is
/// called, it runs all extractors on the response text, evaluates which
/// constraints pass or fail, and returns a `PipelineResult` with the full
/// decomposition.
///
/// The default constructor registers `ArithmeticExtractor` and
/// `LogicExtractor` via `AutoExtractor`. Additional extractors can be
/// added at construction time.
///
/// Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
pub struct VerifyPipeline {
    /// The extractor used to pull constraints from response text.
    extractor: Box<dyn ConstraintExtractor>,
}

impl VerifyPipeline {
    /// Create a new VerifyPipeline with the given extractor.
    pub fn with_extractor(extractor: Box<dyn ConstraintExtractor>) -> Self {
        Self { extractor }
    }

    /// Verify a response by extracting and checking constraints.
    ///
    /// ## For Engineers
    ///
    /// This is the core verification path, ported from Python's
    /// `VerifyRepairPipeline.verify()`:
    ///
    /// 1. Extract constraints from `response` text using the configured
    ///    extractor (AutoExtractor by default).
    /// 2. For each constraint, check the `verified` field:
    ///    - `Some(true)` = satisfied
    ///    - `Some(false)` = violated (added to violations list)
    ///    - `None` = indeterminate (not counted as violation)
    /// 3. Compute total energy as count of violations (simple penalty).
    /// 4. Generate a `VerificationCertificate` recording the result.
    ///
    /// The `question` parameter is included for context/logging but does
    /// not affect extraction (constraints are extracted from `response`).
    ///
    /// Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003
    pub fn verify(&self, _question: &str, response: &str) -> PipelineResult {
        self.verify_with_domain(_question, response, None)
    }

    /// Verify with an optional domain filter for constraint extraction.
    pub fn verify_with_domain(
        &self,
        _question: &str,
        response: &str,
        domain: Option<&str>,
    ) -> PipelineResult {
        // Step 1: Extract constraints from response text.
        let constraints = self.extractor.extract(response, domain);

        // Step 2: Classify satisfied/violated.
        let mut violations = Vec::new();
        for c in &constraints {
            if c.verified == Some(false) {
                violations.push(c.clone());
            }
        }

        // Step 3: Compute energy (simple penalty: 1.0 per violation).
        let energy = violations.len() as f64;
        let verified = violations.is_empty();

        // Step 4: Build certificate.
        let timestamp = Self::iso_timestamp();
        let certificate = Self::build_certificate(&constraints, &violations, energy, &timestamp);

        PipelineResult {
            verified,
            constraints,
            energy,
            violations,
            certificate,
        }
    }

    /// Extract constraints from text without verification (convenience method).
    pub fn extract_constraints(&self, text: &str, domain: Option<&str>) -> Vec<ConstraintResult> {
        self.extractor.extract(text, domain)
    }

    /// Build a VerificationCertificate from pipeline results.
    fn build_certificate(
        constraints: &[ConstraintResult],
        violations: &[ConstraintResult],
        energy: f64,
        timestamp: &str,
    ) -> VerificationCertificate {
        use crate::verify::CertificateConstraint;

        let cert_constraints: Vec<CertificateConstraint> = constraints
            .iter()
            .map(|c| {
                let satisfied = c.verified.unwrap_or(true);
                let e = if satisfied { 0.0 } else { 1.0 };
                CertificateConstraint {
                    name: c.description.clone(),
                    energy: e as carnot_core::Float,
                    weighted_energy: e as carnot_core::Float,
                    satisfied,
                }
            })
            .collect();

        let num_satisfied = cert_constraints.iter().filter(|c| c.satisfied).count();
        let failing_constraints: Vec<String> = violations.iter().map(|v| v.description.clone()).collect();

        let status = if violations.is_empty() {
            CertificateStatus::Verified
        } else {
            CertificateStatus::Violated
        };

        VerificationCertificate {
            timestamp: timestamp.to_string(),
            status,
            total_energy: energy as carnot_core::Float,
            num_constraints: constraints.len(),
            num_satisfied,
            num_violated: violations.len(),
            constraints: cert_constraints,
            failing_constraints,
        }
    }

    /// Get current UTC timestamp in ISO 8601 format.
    fn iso_timestamp() -> String {
        // Use a simple approach without pulling in chrono.
        // In production this would use chrono or time crate.
        // For now, use a fixed format that's deterministic for testing.
        use std::time::{SystemTime, UNIX_EPOCH};
        let dur = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default();
        let secs = dur.as_secs();
        // Convert epoch seconds to ISO 8601 (approximate, no leap seconds).
        let days = secs / 86400;
        let time_of_day = secs % 86400;
        let hours = time_of_day / 3600;
        let minutes = (time_of_day % 3600) / 60;
        let seconds = time_of_day % 60;

        // Compute year/month/day from days since epoch (1970-01-01).
        let (year, month, day) = epoch_days_to_ymd(days);

        format!(
            "{year:04}-{month:02}-{day:02}T{hours:02}:{minutes:02}:{seconds:02}Z"
        )
    }
}

/// Convert days since Unix epoch to (year, month, day).
/// Simple civil calendar conversion (handles leap years correctly).
fn epoch_days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Algorithm from Howard Hinnant's date library (public domain).
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

impl Default for VerifyPipeline {
    fn default() -> Self {
        Self {
            extractor: Box::new(AutoExtractor::new()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_verify_correct_arithmetic() {
        // REQ-VERIFY-003: Correct arithmetic passes verification.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify("What is 47 + 28?", "The answer is 47 + 28 = 75.");
        assert!(result.verified);
        assert_eq!(result.violations.len(), 0);
        assert!(result.energy < 1e-10);
        assert!(result.certificate.is_verified());
    }

    #[test]
    fn test_verify_wrong_arithmetic() {
        // REQ-VERIFY-003: Wrong arithmetic fails verification.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify("What is 47 + 28?", "The answer is 47 + 28 = 76.");
        assert!(!result.verified);
        assert_eq!(result.violations.len(), 1);
        assert!(result.energy > 0.0);
        assert!(!result.certificate.is_verified());
    }

    #[test]
    fn test_verify_multiple_constraints() {
        // REQ-VERIFY-003: Multiple constraints verified together.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify(
            "Compute",
            "First: 2 + 3 = 5. Then: 10 - 4 = 7.",
        );
        // 2+3=5 is correct, 10-4=7 is wrong (should be 6).
        assert!(!result.verified);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.constraints.len(), 2);
    }

    #[test]
    fn test_verify_no_constraints() {
        // REQ-VERIFY-003: Text with no constraints is vacuously verified.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify("Tell me a joke", "Why did the chicken cross the road?");
        assert!(result.verified);
        assert!(result.constraints.is_empty());
    }

    #[test]
    fn test_verify_with_logic() {
        // REQ-VERIFY-003: Logic constraints are extracted (structural, not verified).
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify(
            "Explain weather",
            "If it rains, then the ground is wet. 2 + 2 = 4.",
        );
        // Logic constraint has verified=None (structural), arithmetic is correct.
        assert!(result.verified);
        assert!(result.constraints.len() >= 2);
    }

    #[test]
    fn test_verify_domain_filter() {
        // SCENARIO-VERIFY-002: Domain filter restricts extraction.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify_with_domain(
            "Mixed",
            "2 + 3 = 5. If A then B.",
            Some("arithmetic"),
        );
        assert!(result
            .constraints
            .iter()
            .all(|c| c.constraint_type == "arithmetic"));
    }

    #[test]
    fn test_extract_constraints_convenience() {
        // REQ-VERIFY-001: extract_constraints returns constraints without verification.
        let pipeline = VerifyPipeline::default();
        let constraints = pipeline.extract_constraints("2 + 3 = 5", None);
        assert_eq!(constraints.len(), 1);
        assert_eq!(constraints[0].constraint_type, "arithmetic");
    }

    #[test]
    fn test_certificate_fields() {
        // REQ-VERIFY-003: Certificate has correct field values.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify("Test", "2 + 3 = 5. 10 - 4 = 7.");
        let cert = &result.certificate;
        assert_eq!(cert.num_constraints, 2);
        assert_eq!(cert.num_satisfied, 1);
        assert_eq!(cert.num_violated, 1);
        assert_eq!(cert.failing_constraints.len(), 1);
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        // REQ-VERIFY-003: Pipeline certificates serialize correctly.
        let pipeline = VerifyPipeline::default();
        let result = pipeline.verify("Test", "2 + 3 = 5.");
        let json = result.certificate.to_json().unwrap();
        let restored = VerificationCertificate::from_json(&json).unwrap();
        assert_eq!(result.certificate.status, restored.status);
        assert_eq!(result.certificate.num_constraints, restored.num_constraints);
    }

    #[test]
    fn test_epoch_days_to_ymd() {
        // Verify our date conversion for known dates.
        // 2026-04-09 = day 20552 since epoch
        assert_eq!(epoch_days_to_ymd(0), (1970, 1, 1));
        // 2000-01-01 = day 10957
        assert_eq!(epoch_days_to_ymd(10957), (2000, 1, 1));
    }
}
