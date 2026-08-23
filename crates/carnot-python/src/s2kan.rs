use carnot_core::Float;
use carnot_kan::s2kan::{S2KANConfig, S2KANLayer, S2KANParams};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "RustS2KANLayer")]
pub struct PyS2KANLayer {
    inner: S2KANLayer,
}

#[pymethods]
impl PyS2KANLayer {
    #[new]
    #[pyo3(signature = (input_dim, temperature, gate_logits))]
    fn new(
        input_dim: usize,
        temperature: Float,
        gate_logits: PyReadonlyArray2<Float>,
    ) -> PyResult<Self> {
        let config = S2KANConfig {
            input_dim,
            temperature,
        };
        let params = S2KANParams {
            gate_logits: gate_logits.as_array().to_owned(),
        };
        Ok(Self {
            inner: S2KANLayer::new(config, params),
        })
    }

    fn forward<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<Float>,
    ) -> Bound<'py, PyArray2<Float>> {
        let result = self.inner.forward(&x.as_array());
        result.into_pyarray_bound(py)
    }
}

pub fn register_s2kan_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyS2KANLayer>()?;
    Ok(())
}
