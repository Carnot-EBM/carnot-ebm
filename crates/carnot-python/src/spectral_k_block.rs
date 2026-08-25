//! PyO3 boundary for the exact bounded Ising block heat-bath kernel.
//!
//! The boundary preserves the Rust seed and work counters. It does not provide
//! a Python fallback because parity failures must remain visible.
//!
//! Spec: REQ-RUSTPY-6612, SCENARIO-RUSTPY-6612-MATCHED-CHAIN-PARITY

use carnot_samplers::spectral_k_block::{
    SpectralKBlockConfig, SpectralKBlockCore, SpectralKBlockState,
};
use ndarray::Array2;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_error(error: String) -> PyErr {
    PyValueError::new_err(error)
}

/// Validated Ising model and complete block membership.
#[pyclass(name = "RustSpectralKBlockConfig")]
#[derive(Clone)]
pub struct PySpectralKBlockConfig {
    inner: SpectralKBlockConfig,
}

#[pymethods]
impl PySpectralKBlockConfig {
    #[new]
    fn new(
        couplings: Vec<Vec<f64>>,
        fields: Vec<f64>,
        temperature: f64,
        blocks: Vec<Vec<usize>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralKBlockConfig::new(couplings, fields, temperature, blocks)
                .map_err(value_error)?,
        })
    }
}

/// Portable seeded state for restart and exact parity checks.
#[pyclass(name = "RustSpectralKBlockState")]
#[derive(Clone)]
pub struct PySpectralKBlockState {
    inner: SpectralKBlockState,
}

#[pymethods]
impl PySpectralKBlockState {
    #[new]
    #[pyo3(signature = (spins, rng_state, transition=0, spins_updated=0))]
    fn new(
        spins: Vec<i8>,
        rng_state: u64,
        transition: usize,
        spins_updated: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: SpectralKBlockState::new(spins, rng_state, transition, spins_updated)
                .map_err(value_error)?,
        })
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        state_to_dict(py, &self.inner)
    }
}

/// Rust block heat-bath core exposed without algorithm substitution.
#[pyclass(name = "RustSpectralKBlockCore")]
pub struct PySpectralKBlockCore {
    inner: SpectralKBlockCore,
}

#[pymethods]
impl PySpectralKBlockCore {
    #[new]
    fn new(config: &PySpectralKBlockConfig) -> Self {
        Self {
            inner: SpectralKBlockCore::new(config.inner.clone()),
        }
    }

    fn energy(&self, spins: Vec<i8>) -> PyResult<f64> {
        self.inner.energy(&spins).map_err(value_error)
    }

    fn run_chain<'py>(
        &self,
        py: Python<'py>,
        state: &PySpectralKBlockState,
        burn_in: usize,
        retained_samples: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let outcome = self
            .inner
            .run_chain(&state.inner, burn_in, retained_samples)
            .map_err(value_error)?;
        let samples = Array2::from_shape_vec(
            (retained_samples, self.inner.config.n_spins()),
            outcome.samples,
        )
        .map_err(|error| PyValueError::new_err(format!("sample shape invalid: {error}")))?;
        let result = PyDict::new(py);
        result.set_item("samples", PyArray2::from_owned_array(py, samples))?;
        result.set_item("final_state", state_to_dict(py, &outcome.final_state)?)?;
        result.set_item("transitions", outcome.transitions)?;
        result.set_item("spins_updated", outcome.spins_updated)?;
        Ok(result)
    }
}

fn state_to_dict<'py>(
    py: Python<'py>,
    state: &SpectralKBlockState,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("spins", state.spins.clone())?;
    result.set_item("rng_state", state.rng_state)?;
    result.set_item("transition", state.transition)?;
    result.set_item("spins_updated", state.spins_updated)?;
    Ok(result)
}

pub fn register_spectral_k_block_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "spectral_k_block")?;
    module.add_class::<PySpectralKBlockConfig>()?;
    module.add_class::<PySpectralKBlockState>()?;
    module.add_class::<PySpectralKBlockCore>()?;
    parent.add_submodule(&module)?;

    parent.add_class::<PySpectralKBlockConfig>()?;
    parent.add_class::<PySpectralKBlockState>()?;
    parent.add_class::<PySpectralKBlockCore>()?;
    Ok(())
}
