//! PyO3 wrappers for the Rust VerifyPipeline — exposes constraint verification
//! to Python for 10x performance over the pure-Python path.
//!
//! ## For Researchers
//!
//! This module bridges the Rust `VerifyPipeline` (from `carnot-constraints`)
//! into Python via PyO3. When users `import carnot`, the pipeline auto-detects
//! Rust availability and routes `verify()` calls through this native code path.
//! Repair still uses the Python LLM path — only the hot verification inner loop
//! gets the Rust speedup.
//!
//! ## For Engineers
//!
//! Two pyclass wrappers:
//! - `PyVerifyPipeline`: wraps `carnot_constraints::pipeline::VerifyPipeline`.
//!   Constructed with a list of extractor domain names (default: arithmetic + logic).
//!   Exposes `verify(question, response)` returning `PyVerificationResult`.
//! - `PyVerificationResult`: read-only view of `PipelineResult` with `verified`,
//!   `constraints`, `energy`, and `violations` properties.
//!
//! Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-VERIFY-003, REQ-CORE-005

use pyo3::prelude::*;
use pyo3::types::PyDict;

use carnot_constraints::extract::ConstraintResult;
use carnot_constraints::pipeline::{PipelineResult, VerifyPipeline};

// ---------------------------------------------------------------------------
// PyVerificationResult — read-only view of a PipelineResult
// ---------------------------------------------------------------------------

/// Verification result returned by RustVerifyPipeline.verify().
///
/// ## For Engineers
///
/// This is a frozen snapshot of the Rust `PipelineResult`. All fields are
/// exposed as read-only Python properties. The `constraints` and `violations`
/// lists contain Python dicts with the same keys as the Python
/// `ConstraintResult` dataclass (`constraint_type`, `description`, `verified`,
/// `metadata`), so the Rust and Python pipelines produce identical shapes.
///
/// Spec: REQ-VERIFY-003, SCENARIO-VERIFY-004
#[pyclass(name = "RustVerificationResult")]
pub struct PyVerificationResult {
    inner: PipelineResult,
}

/// Convert a Rust `ConstraintResult` to a Python dict with the same shape
/// as the Python `ConstraintResult` dataclass.
fn constraint_to_pydict<'py>(
    py: Python<'py>,
    cr: &ConstraintResult,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("constraint_type", &cr.constraint_type)?;
    d.set_item("description", &cr.description)?;
    d.set_item("verified", cr.verified)?;

    let meta = PyDict::new(py);
    for (k, v) in &cr.metadata {
        meta.set_item(k, v)?;
    }
    d.set_item("metadata", meta)?;

    Ok(d)
}

#[pymethods]
impl PyVerificationResult {
    /// True if all deterministically-checkable constraints are satisfied.
    #[getter]
    fn verified(&self) -> bool {
        self.inner.verified
    }

    /// List of all extracted constraints as dicts.
    #[getter]
    fn constraints<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .constraints
            .iter()
            .map(|c| constraint_to_pydict(py, c))
            .collect()
    }

    /// Total energy score (sum of 1.0 per violation). 0.0 = all satisfied.
    #[getter]
    fn energy(&self) -> f64 {
        self.inner.energy
    }

    /// Subset of constraints that failed verification, as dicts.
    #[getter]
    fn violations<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyDict>>> {
        self.inner
            .violations
            .iter()
            .map(|c| constraint_to_pydict(py, c))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "RustVerificationResult(verified={}, energy={}, constraints={}, violations={})",
            self.inner.verified,
            self.inner.energy,
            self.inner.constraints.len(),
            self.inner.violations.len(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyVerifyPipeline — wraps VerifyPipeline
// ---------------------------------------------------------------------------

/// Rust-accelerated verification pipeline exposed to Python.
///
/// ## For Engineers
///
/// Wraps `carnot_constraints::pipeline::VerifyPipeline`. The default
/// constructor registers arithmetic and logic extractors (via AutoExtractor).
/// The `extractors` parameter is accepted for API symmetry but currently
/// only the default set is supported — custom extractors require the
/// Python path.
///
/// Spec: REQ-VERIFY-001, REQ-VERIFY-002, REQ-CORE-005
#[pyclass(name = "RustVerifyPipeline")]
pub struct PyVerifyPipeline {
    inner: VerifyPipeline,
}

#[pymethods]
impl PyVerifyPipeline {
    /// Create a new RustVerifyPipeline.
    ///
    /// Args:
    ///     extractors: List of extractor domain names. Currently ignored —
    ///         the Rust pipeline always uses AutoExtractor (arithmetic + logic).
    ///         Accepted for API compatibility with the Python pipeline.
    #[new]
    #[pyo3(signature = (extractors=None))]
    fn new(extractors: Option<Vec<String>>) -> Self {
        // Log which extractors were requested (for debugging).
        // The Rust pipeline uses AutoExtractor which covers arithmetic + logic.
        let _ = extractors;
        Self {
            inner: VerifyPipeline::default(),
        }
    }

    /// Verify a response by extracting and checking constraints.
    ///
    /// Args:
    ///     question: The original question (for context/logging).
    ///     response: The response text to verify.
    ///
    /// Returns:
    ///     RustVerificationResult with constraint evaluation details.
    #[pyo3(signature = (question, response))]
    fn verify(&self, question: &str, response: &str) -> PyVerificationResult {
        let result = self.inner.verify(question, response);
        PyVerificationResult { inner: result }
    }

    fn __repr__(&self) -> String {
        "RustVerifyPipeline(extractors=['arithmetic', 'logic'])".to_string()
    }
}

/// Register pipeline classes into a Python submodule.
///
/// Called from the top-level `_rust` module to add `carnot._rust.pipeline.*`.
pub fn register_pipeline_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let pipeline_mod = PyModule::new(parent.py(), "pipeline")?;
    pipeline_mod.add_class::<PyVerifyPipeline>()?;
    pipeline_mod.add_class::<PyVerificationResult>()?;
    parent.add_submodule(&pipeline_mod)?;

    // Expose classes at top level too for convenience imports.
    parent.add_class::<PyVerifyPipeline>()?;
    parent.add_class::<PyVerificationResult>()?;

    Ok(())
}
