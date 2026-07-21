//! PyO3 bindings for the promoted one-axis corrected-cDLS replica-exchange core.
//!
//! The wrapper deliberately exposes only deterministic diagnostics and
//! checkpoint/state operations needed by Exp5714 parity. It does not expose or
//! implement the retired two-axis penalty-exchange path.
//!
//! Spec: REQ-SAMPLE-5714, SCENARIO-SAMPLE-5714

use carnot_samplers::one_axis_tempering::{
    CorrectedStepOutcome, OneAxisTemperingConfig, OneAxisTemperingCore, OneAxisTemperingState,
    SwapOutcome,
};
use ndarray::Array2;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

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

fn state_to_checkpoint<'py>(
    py: Python<'py>,
    state: &OneAxisTemperingState,
) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("states", state.states.clone())?;
    d.set_item("labels", state.labels.clone())?;
    d.set_item("rng_state", state.rng_state)?;
    d.set_item("sweep", state.sweep)?;
    Ok(d)
}

fn next_uniform(rng_state: &mut u64) -> f64 {
    *rng_state = rng_state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let bits = *rng_state >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn draw_uniforms(rng_state: &mut u64, count: usize) -> Vec<f64> {
    (0..count).map(|_| next_uniform(rng_state)).collect()
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

    #[pyo3(signature = (states, labels, rng_state, sweep, burn_in_sweeps, n_samples))]
    fn run_sweeps<'py>(
        &self,
        py: Python<'py>,
        states: Vec<Vec<i8>>,
        labels: Vec<usize>,
        rng_state: u64,
        sweep: usize,
        burn_in_sweeps: usize,
        n_samples: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if n_samples == 0 {
            return Err(PyValueError::new_err("n_samples must be positive"));
        }
        let mut state =
            OneAxisTemperingState::new(states, labels, rng_state, sweep).map_err(value_error)?;
        let total_sweeps = burn_in_sweeps
            .checked_add(n_samples)
            .ok_or_else(|| PyValueError::new_err("sweep count overflow"))?;
        let replica_count = self.inner.config.beta_ladder.len();
        let n_spins = self.inner.config.n_spins();
        let decision_log = PyList::empty(py);
        let mut samples_spin: Vec<Vec<i8>> = Vec::with_capacity(n_samples);

        for local_sweep in 0..total_sweeps {
            let completed_sweep = state.sweep + 1;
            for physical_index in 0..replica_count {
                let beta_label = state.labels[physical_index];
                let beta = self.inner.config.beta_ladder[beta_label];
                let before = state.states[physical_index].clone();
                let uniforms = draw_uniforms(&mut state.rng_state, n_spins + 1);
                let outcome = self
                    .inner
                    .corrected_step(&before, beta, &uniforms)
                    .map_err(value_error)?;
                let after = outcome.state.clone();
                state.states[physical_index] = after.clone();

                let event = PyDict::new(py);
                event.set_item("kind", "within")?;
                event.set_item("sweep", completed_sweep)?;
                event.set_item("physical_index", physical_index)?;
                event.set_item("beta_label", beta_label)?;
                event.set_item("beta", beta)?;
                event.set_item("uniforms", uniforms)?;
                event.set_item("state_before", before)?;
                event.set_item("state_after", after)?;
                event.set_item("proposed_state", outcome.proposed_state)?;
                event.set_item("current_energy", outcome.current_energy)?;
                event.set_item("proposed_energy", outcome.proposed_energy)?;
                event.set_item("proposal_log_forward", outcome.proposal_log_forward)?;
                event.set_item("proposal_log_reverse", outcome.proposal_log_reverse)?;
                event.set_item("log_acceptance", outcome.log_acceptance)?;
                event.set_item("accepted", outcome.accepted)?;
                decision_log.append(event)?;
            }

            for left in 0..(replica_count - 1) {
                let before_labels = state.labels.clone();
                let uniform = next_uniform(&mut state.rng_state);
                let label_pair = vec![left, left + 1];
                let outcome = self
                    .inner
                    .swap_decision(&state.states, &state.labels, &label_pair, uniform)
                    .map_err(value_error)?;
                state.labels = outcome.labels.clone();

                let event = PyDict::new(py);
                event.set_item("kind", "swap")?;
                event.set_item("sweep", completed_sweep)?;
                event.set_item("label_pair", label_pair)?;
                event.set_item("uniform", uniform)?;
                event.set_item("labels_before", before_labels)?;
                event.set_item("labels_after", state.labels.clone())?;
                event.set_item("proposed_labels", outcome.proposed_labels)?;
                event.set_item("log_ratio", outcome.log_ratio)?;
                event.set_item("acceptance_probability", outcome.acceptance_probability)?;
                event.set_item("accepted", outcome.accepted)?;
                decision_log.append(event)?;
            }

            state.sweep = completed_sweep;
            if local_sweep >= burn_in_sweeps {
                let cold_label = replica_count - 1;
                let position = state
                    .labels
                    .iter()
                    .position(|label| *label == cold_label)
                    .ok_or_else(|| PyValueError::new_err("cold label missing from state"))?;
                samples_spin.push(state.states[position].clone());
            }
        }

        let d = PyDict::new(py);
        d.set_item("samples_spin", samples_spin)?;
        d.set_item("decision_log", decision_log)?;
        d.set_item("final_state", state_to_checkpoint(py, &state)?)?;
        Ok(d)
    }

    #[pyo3(signature = (states, labels, rng_state, sweep, burn_in_sweeps, n_samples))]
    fn run_sweeps_compact<'py>(
        &self,
        py: Python<'py>,
        states: Vec<Vec<i8>>,
        labels: Vec<usize>,
        rng_state: u64,
        sweep: usize,
        burn_in_sweeps: usize,
        n_samples: usize,
    ) -> PyResult<Bound<'py, PyDict>> {
        if n_samples == 0 {
            return Err(PyValueError::new_err("n_samples must be positive"));
        }
        let n_spins = self.inner.config.n_spins();
        let state =
            OneAxisTemperingState::new(states, labels, rng_state, sweep).map_err(value_error)?;
        let outcome = self
            .inner
            .run_compact_sweeps(&state, burn_in_sweeps, n_samples)
            .map_err(value_error)?;
        let samples = Array2::from_shape_vec((n_samples, n_spins), outcome.samples_spin)
            .map_err(|err| PyValueError::new_err(format!("compact sample shape invalid: {err}")))?;

        let counters = PyDict::new(py);
        counters.set_item(
            "rust_per_sample_heap_allocations",
            outcome.counters.rust_per_sample_heap_allocations,
        )?;
        counters.set_item(
            "workspace_allocations",
            outcome.counters.workspace_allocations,
        )?;
        counters.set_item("output_allocations", outcome.counters.output_allocations)?;
        counters.set_item(
            "total_corrected_transitions",
            outcome.counters.total_corrected_transitions,
        )?;
        counters.set_item("total_swap_attempts", outcome.counters.total_swap_attempts)?;

        let buffer_reuse = PyDict::new(py);
        buffer_reuse.set_item(
            "contiguous_samples",
            outcome.buffer_reuse.contiguous_samples,
        )?;
        buffer_reuse.set_item("workspace_reused", outcome.buffer_reuse.workspace_reused)?;
        buffer_reuse.set_item(
            "per_sample_heap_buffers",
            outcome.buffer_reuse.per_sample_heap_buffers,
        )?;

        let worker_pool = PyDict::new(py);
        worker_pool.set_item("fixed_worker_count", outcome.worker_pool.fixed_worker_count)?;
        worker_pool.set_item(
            "dynamic_per_sample_workers",
            outcome.worker_pool.dynamic_per_sample_workers,
        )?;

        let d = PyDict::new(py);
        d.set_item("samples_spin", PyArray2::from_owned_array(py, samples))?;
        d.set_item(
            "final_state",
            state_to_checkpoint(py, &outcome.final_state)?,
        )?;
        d.set_item("allocation_counters", counters)?;
        d.set_item("buffer_reuse_receipt", buffer_reuse)?;
        d.set_item("worker_pool_receipt", worker_pool)?;
        Ok(d)
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
