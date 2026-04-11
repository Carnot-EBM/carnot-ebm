//! # carnot-constraints: Built-in constraint types and verification certificates
//!
//! ## For Researchers
//!
//! Provides reusable constraint primitives (bound, equality, Ising-based) that
//! implement `carnot_core::verify::ConstraintTerm`. Also provides
//! `VerificationCertificate` — a serializable, timestamped proof that a
//! configuration satisfies a set of constraints.
//!
//! ## For Engineers
//!
//! This crate saves you from writing the same constraint boilerplate over and
//! over. Instead of implementing `ConstraintTerm` from scratch every time you
//! need "value must be in range [lo, hi]", just use `BoundConstraint`.
//!
//! **Built-in constraints:**
//! - [`BoundConstraint`] — value at index `i` must be in `[lo, hi]`
//! - [`EqualityConstraint`] — value at index `i` must equal `target` (with tolerance)
//! - [`IsingConstraint`] — wraps a `carnot_ising::IsingModel` as a constraint term,
//!   useful for encoding SAT/logic problems as pairwise energy
//!
//! **Verification certificates:**
//! - [`VerificationCertificate`] — a serializable record proving that a specific
//!   configuration was verified against a specific set of constraints at a given time
//!
//! Re-exports core verification types from `carnot_core::verify` for convenience.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-VERIFY-004

pub mod constraint;
pub mod extract;
pub mod pipeline;
pub mod verify;

// Re-export core verification types so users don't need to depend on carnot-core directly.
pub use carnot_core::verify::{
    ComposedEnergy, ConstraintReport, ConstraintTerm, Verdict, VerificationResult,
};

pub use constraint::{BoundConstraint, EqualityConstraint, IsingConstraint};
pub use extract::{
    ArithmeticExtractor, AutoExtractor, ConstraintExtractor, ConstraintResult, LogicExtractor,
};
pub use pipeline::{PipelineResult, VerifyPipeline};
pub use verify::VerificationCertificate;
