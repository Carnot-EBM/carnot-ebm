//! REQ-VERIFY-8533: copied numeric operands enter a separately versioned kernel.
use carnot_core::nine_input_8533::residuals;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn nine_input_residual_8533(c: Vec<f64>, x: Vec<Vec<f64>>) -> PyResult<Vec<f64>> {
    residuals(&c, &x).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(nine_input_residual_8533, m)?)?;
    Ok(())
}
