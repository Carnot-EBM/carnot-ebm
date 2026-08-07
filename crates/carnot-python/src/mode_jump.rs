//! PyO3 bindings for the fixed Exp6166/Exp6180 mode-jump sampler.
//!
//! Spec: REQ-SAMPLE-6194, REQ-RUSTPY-6194,
//! SCENARIO-RUSTPY-6194-BOUNDARY-PARITY

use carnot_samplers::mode_jump::{
    ModeJumpConfig, ModeJumpCore, ModeJumpRunSummary, ModeJumpState, ModeJumpStepOutcome,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn value_error(error: String) -> PyErr {
    PyValueError::new_err(error)
}

fn state_to_dict<'py>(py: Python<'py>, state: &ModeJumpState) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("current_label", state.current_label.clone())?;
    d.set_item("rng_state", state.rng_state)?;
    d.set_item("step", state.step)?;
    d.set_item("accepted_count", state.accepted_count)?;
    Ok(d)
}

fn state_from_dict(snapshot: &Bound<'_, PyDict>) -> PyResult<ModeJumpState> {
    let current_label = snapshot
        .get_item("current_label")?
        .ok_or_else(|| PyValueError::new_err("snapshot missing current_label"))?
        .extract::<String>()?;
    let rng_state = snapshot
        .get_item("rng_state")?
        .ok_or_else(|| PyValueError::new_err("snapshot missing rng_state"))?
        .extract::<u64>()?;
    let step = snapshot
        .get_item("step")?
        .ok_or_else(|| PyValueError::new_err("snapshot missing step"))?
        .extract::<usize>()?;
    let accepted_count = snapshot
        .get_item("accepted_count")?
        .ok_or_else(|| PyValueError::new_err("snapshot missing accepted_count"))?
        .extract::<usize>()?;
    ModeJumpState::new(current_label, rng_state, step, accepted_count).map_err(value_error)
}

fn step_to_dict<'py>(
    py: Python<'py>,
    outcome: ModeJumpStepOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("state_before", state_to_dict(py, &outcome.state_before)?)?;
    d.set_item("state_after", state_to_dict(py, &outcome.state)?)?;
    d.set_item("proposal_uniform", outcome.proposal_uniform)?;
    d.set_item("proposed_label", outcome.proposed_label)?;
    d.set_item("acceptance_uniform", outcome.acceptance_uniform)?;
    d.set_item("current_energy", outcome.current_energy)?;
    d.set_item("proposed_energy", outcome.proposed_energy)?;
    d.set_item("proposal_log_forward", outcome.proposal_log_forward)?;
    d.set_item("proposal_log_reverse", outcome.proposal_log_reverse)?;
    d.set_item("log_acceptance", outcome.log_acceptance)?;
    d.set_item("acceptance_probability", outcome.acceptance_probability)?;
    d.set_item("accepted", outcome.accepted)?;
    d.set_item("rng_state_after", outcome.state.rng_state)?;
    Ok(d)
}

fn summary_to_dict<'py>(
    py: Python<'py>,
    summary: ModeJumpRunSummary,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("sample_count", summary.sample_count)?;
    d.set_item("burn_in", summary.burn_in)?;
    let rows = PyList::empty(py);
    for row in summary.frequencies {
        let item = PyDict::new(py);
        item.set_item("label", row.label)?;
        item.set_item("count", row.count)?;
        item.set_item("frequency", row.frequency)?;
        item.set_item("target_probability", row.target_probability)?;
        rows.append(item)?;
    }
    d.set_item("frequencies", rows)?;
    d.set_item(
        "total_variation_to_target",
        summary.total_variation_to_target,
    )?;
    d.set_item("kl_target_to_empirical", summary.kl_target_to_empirical)?;
    d.set_item("accepted_count", summary.accepted_count)?;
    d.set_item("attempted_count", summary.attempted_count)?;
    d.set_item("acceptance_rate", summary.acceptance_rate)?;
    d.set_item("lag1_autocorrelation", summary.lag1_autocorrelation)?;
    d.set_item(
        "integrated_autocorrelation_time",
        summary.integrated_autocorrelation_time,
    )?;
    d.set_item("effective_sample_size", summary.effective_sample_size)?;
    d.set_item("final_state", state_to_dict(py, &summary.final_state)?)?;
    d.set_item("serialized_final_state", summary.final_state.serialize())?;
    Ok(d)
}

/// Validated finite categorical target and proposal table.
#[pyclass(name = "RustModeJumpConfig")]
#[derive(Clone)]
pub struct PyModeJumpConfig {
    inner: ModeJumpConfig,
}

#[pymethods]
impl PyModeJumpConfig {
    #[new]
    fn new(
        labels: Vec<String>,
        target_probabilities: Vec<f64>,
        proposal_probabilities: Vec<Vec<f64>>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ModeJumpConfig::new(labels, target_probabilities, proposal_probabilities)
                .map_err(value_error)?,
        })
    }
}

/// Serializable seeded mode-jump state.
#[pyclass(name = "RustModeJumpState")]
#[derive(Clone)]
pub struct PyModeJumpState {
    inner: ModeJumpState,
}

#[pymethods]
impl PyModeJumpState {
    #[new]
    #[pyo3(signature = (current_label, rng_state, step=0, accepted_count=0))]
    fn new(
        current_label: String,
        rng_state: u64,
        step: usize,
        accepted_count: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: ModeJumpState::new(current_label, rng_state, step, accepted_count)
                .map_err(value_error)?,
        })
    }

    #[staticmethod]
    fn from_snapshot(snapshot: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            inner: state_from_dict(snapshot)?,
        })
    }

    #[staticmethod]
    fn deserialize(serialized: &str) -> PyResult<Self> {
        Ok(Self {
            inner: ModeJumpState::deserialize(serialized).map_err(value_error)?,
        })
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        state_to_dict(py, &self.inner)
    }

    fn serialize(&self) -> String {
        self.inner.serialize()
    }
}

/// Fixed mode-jump Metropolis-Hastings core.
#[pyclass(name = "RustModeJumpCore")]
pub struct PyModeJumpCore {
    inner: ModeJumpCore,
}

#[pymethods]
impl PyModeJumpCore {
    #[new]
    fn new(config: &PyModeJumpConfig) -> Self {
        Self {
            inner: ModeJumpCore::new(config.inner.clone()),
        }
    }

    fn energy(&self, label: String) -> PyResult<f64> {
        self.inner.energy(&label).map_err(value_error)
    }

    fn proposal_probability(&self, current: String, proposed: String) -> PyResult<f64> {
        self.inner
            .proposal_probability(&current, &proposed)
            .map_err(value_error)
    }

    fn state_from_serialized(&self, serialized: &str) -> PyResult<PyModeJumpState> {
        Ok(PyModeJumpState {
            inner: self
                .inner
                .state_from_serialized(serialized)
                .map_err(value_error)?,
        })
    }

    fn step(&self, state: &PyModeJumpState) -> PyResult<PyModeJumpState> {
        Ok(PyModeJumpState {
            inner: self
                .inner
                .step_trace(&state.inner)
                .map_err(value_error)?
                .state,
        })
    }

    fn step_trace<'py>(
        &self,
        py: Python<'py>,
        state: &PyModeJumpState,
    ) -> PyResult<Bound<'py, PyDict>> {
        let outcome = self.inner.step_trace(&state.inner).map_err(value_error)?;
        step_to_dict(py, outcome)
    }

    #[pyo3(signature = (state, n_steps, burn_in=0))]
    fn run<'py>(
        &self,
        py: Python<'py>,
        state: &PyModeJumpState,
        n_steps: usize,
        burn_in: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        let summary = self
            .inner
            .run(&state.inner, n_steps, burn_in)
            .map_err(value_error)?;
        summary_to_dict(py, summary)
    }
}

pub fn register_mode_jump_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "mode_jump")?;
    module.add_class::<PyModeJumpConfig>()?;
    module.add_class::<PyModeJumpState>()?;
    module.add_class::<PyModeJumpCore>()?;
    parent.add_submodule(&module)?;

    parent.add_class::<PyModeJumpConfig>()?;
    parent.add_class::<PyModeJumpState>()?;
    parent.add_class::<PyModeJumpCore>()?;
    Ok(())
}
