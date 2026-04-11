//! Verification certificates — serializable proof of constraint satisfaction.
//!
//! ## For Researchers
//!
//! A `VerificationCertificate` is a timestamped, serializable record that a
//! specific configuration was verified against a specific set of constraints.
//! It captures the full decomposition (per-constraint energies) and the binary
//! verdict. Certificates can be stored, transmitted, and independently audited.
//!
//! ## For Engineers
//!
//! Think of a `VerificationCertificate` as a "proof of correctness" document.
//! When your system verifies that an AI-generated answer satisfies all constraints,
//! it produces a certificate that says:
//!
//! - **When**: timestamp of verification
//! - **What was checked**: list of constraint names and their energies
//! - **Result**: VERIFIED or VIOLATED (with details on which constraints failed)
//! - **Total energy**: overall "wrongness" score (0 = perfect)
//!
//! Certificates are serializable to JSON, so you can:
//! - Store them in a database for audit trails
//! - Send them over the network to prove correctness to a remote party
//! - Compare certificates over time to track model improvement
//!
//! Spec: REQ-VERIFY-003

use carnot_core::verify::{ComposedEnergy, ConstraintReport, Verdict, VerificationResult};
use carnot_core::Float;
use ndarray::ArrayView1;

/// Status of a verification certificate: VERIFIED (all pass) or VIOLATED (some fail).
///
/// This is a simplified, serializable version of the `Verdict` enum from
/// `carnot_core::verify`. It strips out the failing constraint names (those
/// are available in the per-constraint reports) and just captures the binary
/// outcome, making it easy to store and query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CertificateStatus {
    /// All constraints satisfied. The configuration is verified correct.
    Verified,
    /// One or more constraints violated. See `failing_constraints` for details.
    Violated,
}

/// A single constraint's result within a certificate.
///
/// Serializable version of `ConstraintReport` for inclusion in certificates.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CertificateConstraint {
    /// Human-readable constraint name.
    pub name: String,
    /// Raw (unweighted) energy. 0.0 = satisfied.
    pub energy: Float,
    /// Energy after weighting.
    pub weighted_energy: Float,
    /// Whether this individual constraint was satisfied.
    pub satisfied: bool,
}

impl From<&ConstraintReport> for CertificateConstraint {
    fn from(report: &ConstraintReport) -> Self {
        Self {
            name: report.name.clone(),
            energy: report.energy,
            weighted_energy: report.weighted_energy,
            satisfied: report.satisfied,
        }
    }
}

/// A serializable, timestamped proof that a configuration was verified.
///
/// ## For Researchers
///
/// Captures the complete verification state: timestamp, per-constraint
/// decomposition, total energy, and binary verdict. Designed for reproducible
/// auditing and longitudinal tracking of verification outcomes.
///
/// ## For Engineers
///
/// This is the "receipt" you get after verifying a configuration. It's
/// serializable to JSON, so you can store it, send it, or display it.
///
/// Example JSON output:
/// ```json
/// {
///   "timestamp": "2026-04-09T19:00:00Z",
///   "status": "Verified",
///   "total_energy": 0.0,
///   "num_constraints": 3,
///   "num_satisfied": 3,
///   "num_violated": 0,
///   "constraints": [...],
///   "failing_constraints": []
/// }
/// ```
///
/// Spec: REQ-VERIFY-003
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationCertificate {
    /// ISO 8601 timestamp of when verification was performed.
    pub timestamp: String,
    /// Overall status: Verified or Violated.
    pub status: CertificateStatus,
    /// Sum of all weighted constraint energies.
    pub total_energy: Float,
    /// Total number of constraints checked.
    pub num_constraints: usize,
    /// Number of constraints that passed.
    pub num_satisfied: usize,
    /// Number of constraints that failed.
    pub num_violated: usize,
    /// Per-constraint breakdown.
    pub constraints: Vec<CertificateConstraint>,
    /// Names of failing constraints (empty if status == Verified).
    pub failing_constraints: Vec<String>,
}

impl VerificationCertificate {
    /// Generate a verification certificate by verifying a configuration against
    /// a composed energy function.
    ///
    /// This is the primary API: pass in the constraints and the configuration,
    /// and get back a serializable certificate documenting the result.
    ///
    /// # Arguments
    /// * `composed` - The composed energy function containing all constraints
    /// * `x` - The configuration to verify
    /// * `timestamp` - ISO 8601 timestamp string (caller provides for reproducibility)
    ///
    /// # Returns
    /// A `VerificationCertificate` capturing the full verification state.
    pub fn generate(composed: &ComposedEnergy, x: &ArrayView1<Float>, timestamp: &str) -> Self {
        let result: VerificationResult = composed.verify(x);
        Self::from_result(&result, timestamp)
    }

    /// Create a certificate from an existing `VerificationResult`.
    ///
    /// Useful when you've already computed the verification and want to
    /// wrap it in a certificate without re-computing.
    pub fn from_result(result: &VerificationResult, timestamp: &str) -> Self {
        let constraints: Vec<CertificateConstraint> = result
            .constraints
            .iter()
            .map(CertificateConstraint::from)
            .collect();

        let num_satisfied = constraints.iter().filter(|c| c.satisfied).count();
        let num_violated = constraints.len() - num_satisfied;

        let failing_constraints: Vec<String> = constraints
            .iter()
            .filter(|c| !c.satisfied)
            .map(|c| c.name.clone())
            .collect();

        let status = match &result.verdict {
            Verdict::Verified => CertificateStatus::Verified,
            Verdict::Violated { .. } => CertificateStatus::Violated,
        };

        Self {
            timestamp: timestamp.to_string(),
            status,
            total_energy: result.total_energy,
            num_constraints: constraints.len(),
            num_satisfied,
            num_violated,
            constraints,
            failing_constraints,
        }
    }

    /// Returns true if the certificate records a successful verification.
    pub fn is_verified(&self) -> bool {
        self.status == CertificateStatus::Verified
    }

    /// Serialize the certificate to a JSON string.
    ///
    /// Useful for storing in databases, sending over HTTP, or writing to files.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize a certificate from a JSON string.
    ///
    /// Useful for loading stored certificates for audit or comparison.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraint::{BoundConstraint, EqualityConstraint};
    use carnot_core::verify::ComposedEnergy;
    use ndarray::array;

    #[test]
    fn test_certificate_verified() {
        // REQ-VERIFY-003: certificate for a fully satisfied configuration
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(BoundConstraint::new("x0_range", 0, 0.0, 10.0)),
            1.0,
        );
        composed.add_constraint(
            Box::new(EqualityConstraint::new("x1_target", 1, 5.0, 0.01)),
            1.0,
        );

        let x = array![3.0, 5.0]; // both satisfied
        let cert = VerificationCertificate::generate(&composed, &x.view(), "2026-04-09T19:00:00Z");

        assert!(cert.is_verified());
        assert_eq!(cert.status, CertificateStatus::Verified);
        assert_eq!(cert.num_constraints, 2);
        assert_eq!(cert.num_satisfied, 2);
        assert_eq!(cert.num_violated, 0);
        assert!(cert.failing_constraints.is_empty());
        assert!(cert.total_energy < 1e-6);
    }

    #[test]
    fn test_certificate_violated() {
        // REQ-VERIFY-003: certificate with violations records failing names
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(BoundConstraint::new("x0_range", 0, 5.0, 10.0)),
            1.0,
        );
        composed.add_constraint(
            Box::new(EqualityConstraint::new("x1_target", 1, 5.0, 0.01)),
            1.0,
        );

        let x = array![1.0, 5.0]; // x0 violated (1 < 5), x1 satisfied
        let cert = VerificationCertificate::generate(&composed, &x.view(), "2026-04-09T19:00:00Z");

        assert!(!cert.is_verified());
        assert_eq!(cert.status, CertificateStatus::Violated);
        assert_eq!(cert.num_satisfied, 1);
        assert_eq!(cert.num_violated, 1);
        assert_eq!(cert.failing_constraints, vec!["x0_range"]);
        assert!(cert.total_energy > 0.0);
    }

    #[test]
    fn test_certificate_json_roundtrip() {
        // REQ-VERIFY-003: certificates serialize and deserialize correctly
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(BoundConstraint::new("x0_range", 0, 0.0, 10.0)),
            1.0,
        );

        let x = array![5.0, 0.0];
        let cert = VerificationCertificate::generate(&composed, &x.view(), "2026-04-09T19:00:00Z");

        let json = cert.to_json().unwrap();
        let restored = VerificationCertificate::from_json(&json).unwrap();

        assert_eq!(cert.status, restored.status);
        assert_eq!(cert.num_constraints, restored.num_constraints);
        assert_eq!(cert.num_satisfied, restored.num_satisfied);
        assert!((cert.total_energy - restored.total_energy).abs() < 1e-10);
        assert_eq!(cert.timestamp, restored.timestamp);
    }

    #[test]
    fn test_certificate_from_result() {
        // REQ-VERIFY-003: certificate can be created from existing VerificationResult
        let mut composed = ComposedEnergy::new(2);
        composed.add_constraint(
            Box::new(BoundConstraint::new("x0_range", 0, 0.0, 10.0)),
            1.0,
        );

        let x = array![5.0, 0.0];
        let result = composed.verify(&x.view());
        let cert = VerificationCertificate::from_result(&result, "2026-04-09T19:00:00Z");

        assert!(cert.is_verified());
        assert_eq!(cert.constraints.len(), 1);
        assert_eq!(cert.constraints[0].name, "x0_range");
    }
}
