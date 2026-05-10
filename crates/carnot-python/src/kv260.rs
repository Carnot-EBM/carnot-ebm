//! PyO3 wrapper for the KV260 q=3 Potts sampler binding.
//!
//! The wrapper deliberately does not open `/dev/uio4` during construction.
//! Importing `carnot._rust` must remain safe on laptops and CI machines that
//! do not have the KV260 overlay loaded.  Hardware access happens only when
//! `sample(...)` is called.
//!
//! Spec: REQ-POTTS-008-4

use std::time::Duration;

use carnot_webgpu_gateway::kv260_bindings::{
    Kv260BindingError, Kv260PottsProblem, Kv260PottsSampler, ADDR_ADJ_BASE, ADDR_BETA_FINAL,
    ADDR_BIAS_BASE, ADDR_CONTROL, ADDR_COUPL_BASE, ADDR_SPIN_COUNT, ADDR_SPOUT_BASE, ADDR_STATUS,
    DEFAULT_MAX_DEGREE, DEFAULT_UIO_PATH, STATUS_DONE_MASK,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn binding_error_to_py(error: Kv260BindingError) -> PyErr {
    match error {
        Kv260BindingError::InvalidProblem(_) => {
            pyo3::exceptions::PyValueError::new_err(error.to_string())
        }
        Kv260BindingError::Io(_) | Kv260BindingError::RegisterOutOfRange { .. } => {
            pyo3::exceptions::PyOSError::new_err(error.to_string())
        }
        Kv260BindingError::Timeout { .. } => {
            pyo3::exceptions::PyTimeoutError::new_err(error.to_string())
        }
        Kv260BindingError::InvalidArtifact(_) | Kv260BindingError::Json(_) => {
            pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
        }
    }
}

/// Rust KV260 q=3 Potts sampler exposed to Python.
///
/// The sampler communicates with the Kria generic-UIO driver only when
/// `sample()` is invoked.  This keeps the verification package importable on
/// non-FPGA machines while still giving the Python pipeline a native hardware
/// execution boundary for Exp 1704.
#[pyclass(name = "RustKv260PottsSampler")]
pub struct PyKv260PottsSampler {
    inner: Kv260PottsSampler,
}

#[pymethods]
impl PyKv260PottsSampler {
    #[new]
    #[pyo3(signature = (driver_path=None, timeout_ms=50))]
    fn new(driver_path: Option<String>, timeout_ms: u64) -> Self {
        let path = driver_path.unwrap_or_else(|| DEFAULT_UIO_PATH.to_string());
        Self {
            inner: Kv260PottsSampler::new(path, Duration::from_millis(timeout_ms)),
        }
    }

    #[getter]
    fn driver_path(&self) -> String {
        self.inner.device_path.to_string_lossy().into_owned()
    }

    fn register_map<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("control", ADDR_CONTROL)?;
        d.set_item("status", ADDR_STATUS)?;
        d.set_item("spin_count", ADDR_SPIN_COUNT)?;
        d.set_item("beta_final", ADDR_BETA_FINAL)?;
        d.set_item("bias_base", ADDR_BIAS_BASE)?;
        d.set_item("adjacency_base", ADDR_ADJ_BASE)?;
        d.set_item("coupling_base", ADDR_COUPL_BASE)?;
        d.set_item("spin_output_base", ADDR_SPOUT_BASE)?;
        d.set_item("status_done_mask", STATUS_DONE_MASK)?;
        Ok(d)
    }

    #[pyo3(signature = (n_spins, biases, adjacency, couplings, max_degree=DEFAULT_MAX_DEGREE, beta_fixed=0x40))]
    fn sample(
        &self,
        n_spins: usize,
        biases: Vec<i8>,
        adjacency: Vec<i16>,
        couplings: Vec<i8>,
        max_degree: usize,
        beta_fixed: u8,
    ) -> PyResult<Vec<u8>> {
        let problem = Kv260PottsProblem::new(
            n_spins, max_degree, beta_fixed, biases, adjacency, couplings,
        )
        .map_err(binding_error_to_py)?;
        let sample = self.inner.sample(&problem).map_err(binding_error_to_py)?;
        Ok(sample.states)
    }

    fn __repr__(&self) -> String {
        format!(
            "RustKv260PottsSampler(driver_path='{}')",
            self.driver_path()
        )
    }
}

pub fn register_kv260_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let kv260_mod = PyModule::new(parent.py(), "kv260")?;
    kv260_mod.add_class::<PyKv260PottsSampler>()?;
    parent.add_submodule(&kv260_mod)?;
    parent.add_class::<PyKv260PottsSampler>()?;
    Ok(())
}
