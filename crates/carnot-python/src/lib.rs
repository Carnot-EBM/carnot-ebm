//! carnot-python: PyO3 bindings for the Carnot EBM framework.
//!
//! Spec: REQ-CORE-005, SCENARIO-CORE-005

use pyo3::prelude::*;

/// Carnot EBM framework — Python bindings.
#[pymodule]
fn carnot_python(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    // Model classes and sampler bindings will be added as implementations mature.
    // See REQ-CORE-005 for the full binding specification.
    Ok(())
}
