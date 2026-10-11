//! REQ-VERIFY-8379: this opt-in binding copies only explicit numeric operands.

use carnot_core::direct_spline_8379 as arithmetic;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
fn direct_design_8379(x: Vec<f64>) -> PyResult<Vec<f64>> {
    arithmetic::design(&x).map_err(PyValueError::new_err)
}

#[pyfunction]
fn direct_logits_8379(c: Vec<f64>, x: Vec<Vec<f64>>, temperature: f64) -> PyResult<Vec<f64>> {
    arithmetic::logits(&c, &x, temperature).map_err(PyValueError::new_err)
}

#[pyfunction]
fn direct_update_8379(c: Vec<f64>, x: Vec<f64>, residual: f64) -> PyResult<Vec<f64>> {
    arithmetic::update(&c, &x, residual).map_err(PyValueError::new_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(direct_design_8379, m)?)?;
    m.add_function(wrap_pyfunction!(direct_logits_8379, m)?)?;
    m.add_function(wrap_pyfunction!(direct_update_8379, m)?)?;
    Ok(())
}
