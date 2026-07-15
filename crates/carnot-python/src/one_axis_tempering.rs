//! PyO3 bindings for the promoted one-axis corrected-cDLS replica-exchange core.
//!
//! The wrapper deliberately exposes only deterministic diagnostics and
//! checkpoint/state operations needed by Exp5714 parity. It does not expose or
//! implement the retired two-axis penalty-exchange path.
//!
//! Spec: REQ-SAMPLE-5714, SCENARIO-SAMPLE-5714

use carnot_samplers::one_axis_tempering::{
    CorrectedStepOutcome, OneAxisTemperingConfig, OneAxisTemperingCore,
    OneAxisTemperingState, SwapOutcome,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

fn value_error(error: String) -> PyErr {
    PyValueError::new_err(error)
}

fn corrected_step_to_dict<'py>(
    py: Python<'py>,
    outcome: CorrectedStepOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("state", outcome.state)?;
    d.set_item("proposed_state", outcome.proposed_state)?;
    d.set_item("current_energy", outcome.current_energy)?;
    d.set_item("proposed_energy", outcome.proposed_energy)?;
    d.set_item("proposal_log_forward", outcome.proposal_log_forward)?;
    d.set_item("proposal_log_reverse", outcome.proposal_log_reverse)?;
    d.set_item("log_acceptance", outcome.log_acceptance)?;
    d.set_item("accepted", outcome.accepted)?;
    Ok(d)
}

fn swap_to_dict<'py>(py: Python<'py>, outcome: SwapOutcome) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("labels", outcome.labels)?;
    d.set_item("proposed_labels", outcome.proposed_labels)?;
    d.set_item("log_ratio", outcome.log_ratio)?;
    d.set_item("acceptance_probability", outcome.acceptance_probability)?;
    d.set_item("accepted", outcome.accepted)?;
    Ok(d)
}

/// Validated one-axis tempering configuration.
#[pyclass(name = "RustOneAxisTemperingConfig")]
#[derive(Clone)]
pub struct PyOneAxisTemperingConfig {
    inner: OneAxisTemperingConfig,
}

#[pymethods]
impl PyOneAxisTemperingConfig {
    #[new]
    #[pyo3(signature = (couplings, fields, beta_ladder, proposal_std=0.72, drift_scale=0.17))]
    fn new(
        couplings: Vec<Vec<f64>>,
        fields: Vec<f64>,
        beta_ladder: Vec<f64>,
        proposal_std: f64,
        drift_scale: f64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OneAxisTemperingConfig::new(
                couplings,
                fields,
                beta_ladder,
                proposal_std,
                drift_scale,
            )
            .map_err(value_error)?,
        })
    }
}

/// Serializable one-axis replica state.
#[pyclass(name = "RustOneAxisTemperingState")]
#[derive(Clone)]
pub struct PyOneAxisTemperingState {
    inner: OneAxisTemperingState,
}

#[pymethods]
impl PyOneAxisTemperingState {
    #[new]
    #[pyo3(signature = (states, labels, rng_state, sweep=0))]
    fn new(
        states: Vec<Vec<i8>>,
        labels: Vec<usize>,
        rng_state: u64,
        sweep: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: OneAxisTemperingState::new(states, labels, rng_state, sweep)
                .map_err(value_error)?,
        })
    }

    #[staticmethod]
    fn from_checkpoint(checkpoint: &Bound<'_, PyDict>) -> PyResult<Self> {
        let states = checkpoint
            .get_item("states")?
            .ok_or_else(|| PyValueError::new_err("checkpoint missing states"))?
            .extract::<Vec<Vec<i8>>>()?;
        let labels = checkpoint
            .get_item("labels")?
            .ok_or_else(|| PyValueError::new_err("checkpoint missing labels"))?
            .extract::<Vec<usize>>()?;
        let rng_state = checkpoint
            .get_item("rng_state")?
            .ok_or_else(|| PyValueError::new_err("checkpoint missing rng_state"))?
            .extract::<u64>()?;
        let sweep = checkpoint
            .get_item("sweep")?
            .ok_or_else(|| PyValueError::new_err("checkpoint missing sweep"))?
            .extract::<usize>()?;
        Self::new(states, labels, rng_state, sweep)
            .map_err(|err| PyValueError::new_err(format!("checkpoint invalid: {err}")))
    }

    fn checkpoint<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("states", self.inner.states.clone())?;
        d.set_item("labels", self.inner.labels.clone())?;
        d.set_item("rng_state", self.inner.rng_state)?;
        d.set_item("sweep", self.inner.sweep)?;
        Ok(d)
    }
}

/// Minimal Rust one-axis corrected-cDLS temperature-label exchange core.
#[pyclass(name = "RustOneAxisTemperingCore")]
pub struct PyOneAxisTemperingCore {
    inner: OneAxisTemperingCore,
}

#[pymethods]
impl PyOneAxisTemperingCore {
    #[new]
    fn new(config: &PyOneAxisTemperingConfig) -> Self {
        Self {
            inner: OneAxisTemperingCore::new(config.inner.clone()),
        }
    }

    fn energy(&self, state: Vec<i8>) -> PyResult<f64> {
        self.inner.energy(&state).map_err(value_error)
    }

    fn proposal_log_probability(
        &self,
        source: Vec<i8>,
        target: Vec<i8>,
        beta: f64,
    ) -> PyResult<f64> {
        self.inner
            .proposal_log_probability(&source, &target, beta)
            .map_err(value_error)
    }

    fn corrected_step<'py>(
        &self,
        py: Python<'py>,
        state: Vec<i8>,
        beta: f64,
        uniforms: Vec<f64>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let outcome = self
            .inner
            .corrected_step(&state, beta, &uniforms)
            .map_err(value_error)?;
        corrected_step_to_dict(py, outcome)
    }

    fn swap_log_ratio(
        &self,
        states: Vec<Vec<i8>>,
        labels: Vec<usize>,
        label_pair: Vec<usize>,
    ) -> PyResult<f64> {
        self.inner
            .swap_log_ratio(&states, &labels, &label_pair)
            .map_err(value_error)
    }

    fn swap_decision<'py>(
        &self,
        py: Python<'py>,
        states: Vec<Vec<i8>>,
        labels: Vec<usize>,
        label_pair: Vec<usize>,
        uniform: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let outcome = self
            .inner
            .swap_decision(&states, &labels, &label_pair, uniform)
            .map_err(value_error)?;
        swap_to_dict(py, outcome)
    }

    fn scheduler_trace(&self) -> Vec<String> {
        self.inner.scheduler_trace()
    }

    fn step(&self, state: &PyOneAxisTemperingState) -> PyResult<PyOneAxisTemperingState> {
        Ok(PyOneAxisTemperingState {
            inner: self.inner.step(&state.inner).map_err(value_error)?,
        })
    }

    fn target_state(&self, state: &PyOneAxisTemperingState) -> PyResult<Vec<i8>> {
        self.inner.target_state(&state.inner).map_err(value_error)
    }
}

pub fn register_one_axis_tempering_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "one_axis_tempering")?;
    module.add_class::<PyOneAxisTemperingConfig>()?;
    module.add_class::<PyOneAxisTemperingState>()?;
    module.add_class::<PyOneAxisTemperingCore>()?;
    parent.add_submodule(&module)?;

    parent.add_class::<PyOneAxisTemperingConfig>()?;
    parent.add_class::<PyOneAxisTemperingState>()?;
    parent.add_class::<PyOneAxisTemperingCore>()?;
    Ok(())
}
