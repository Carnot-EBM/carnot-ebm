//! Persistent PyO3 boundary for the fixed-cardinality pair-swap kernel.
//!
//! Python owns the NumPy inputs. Rust copies each result before it returns, so
//! later input changes cannot alter recorded evidence. The object keeps one
//! tape allocation for repeated calls to avoid rebuilding boundary storage.
//!
//! Spec: REQ-RUSTPY-7201, SCENARIO-RUSTPY-7201-PERSISTENT-PARITY

use carnot_samplers::fixed_cardinality::{
    PairSwapChainOutcome, PairSwapConfig, PairSwapCore, PairSwapDraw, PairSwapReplayOutcome,
    PairSwapSeededState, PairSwapStepOutcome, StoredEdge,
};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

fn seeded_state_to_dict<'py>(
    py: Python<'py>,
    state: &PairSwapSeededState,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("spins", state.spins.clone())?;
    result.set_item("rng_state", state.rng_state)?;
    result.set_item("transition", state.transition)?;
    Ok(result)
}

fn seeded_state_from_dict(snapshot: &Bound<'_, PyDict>) -> PyResult<PairSwapSeededState> {
    let spins = snapshot
        .get_item("spins")?
        .ok_or_else(|| value_error("state missing spins"))?
        .extract::<Vec<i8>>()?;
    let rng_state = snapshot
        .get_item("rng_state")?
        .ok_or_else(|| value_error("state missing rng_state"))?
        .extract::<u64>()?;
    let transition = snapshot
        .get_item("transition")?
        .ok_or_else(|| value_error("state missing transition"))?
        .extract::<usize>()?;
    let mut state = PairSwapSeededState::new(spins, rng_state).map_err(value_error)?;
    state.transition = transition;
    Ok(state)
}

fn step_to_dict<'py>(py: Python<'py>, step: &PairSwapStepOutcome) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("state", step.state.clone())?;
    result.set_item("proposed_state", step.proposed_state.clone())?;
    result.set_item("current_energy", step.current_energy)?;
    result.set_item("proposed_energy", step.proposed_energy)?;
    result.set_item("delta_energy", step.delta_energy)?;
    result.set_item("log_acceptance", step.log_acceptance)?;
    result.set_item("accepted", step.accepted)?;
    result.set_item("cardinality", step.cardinality)?;
    Ok(result)
}

fn replay_to_dict<'py>(
    py: Python<'py>,
    outcome: &PairSwapReplayOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("initial_state", outcome.initial_state.clone())?;
    result.set_item("final_state", outcome.final_state.clone())?;
    let steps = PyList::empty(py);
    for step in &outcome.steps {
        steps.append(step_to_dict(py, step)?)?;
    }
    result.set_item("steps", steps)?;
    Ok(result)
}

fn chain_to_dict<'py>(
    py: Python<'py>,
    outcome: &PairSwapChainOutcome,
) -> PyResult<Bound<'py, PyDict>> {
    let result = PyDict::new(py);
    result.set_item("samples", outcome.samples.clone())?;
    result.set_item("energies", outcome.energies.clone())?;
    result.set_item("accepted", outcome.accepted)?;
    result.set_item("attempted", outcome.attempted)?;
    result.set_item("energy_evaluations", outcome.energy_evaluations)?;
    result.set_item(
        "final_state",
        seeded_state_to_dict(py, &outcome.final_state)?,
    )?;
    Ok(result)
}

/// Persistent compiled object for one immutable Ising slice configuration.
#[pyclass(name = "RustFixedCardinalitySampler")]
pub struct PyFixedCardinalitySampler {
    core: PairSwapCore,
    tape_buffer: Vec<PairSwapDraw>,
    call_count: usize,
    max_batch_size: usize,
}

#[pymethods]
impl PyFixedCardinalitySampler {
    #[new]
    fn new(
        edges: Vec<(usize, usize, f64)>,
        fields: Vec<f64>,
        cardinality: usize,
        beta: f64,
    ) -> PyResult<Self> {
        let edges = edges
            .into_iter()
            .map(|(left, right, coupling)| StoredEdge::try_new(left, right, coupling))
            .collect::<Result<Vec<_>, _>>()
            .map_err(value_error)?;
        let config = PairSwapConfig::new(edges, fields, cardinality, beta).map_err(value_error)?;
        Ok(Self {
            core: PairSwapCore::new(config),
            tape_buffer: Vec::new(),
            call_count: 0,
            max_batch_size: 0,
        })
    }

    /// Replay one tape per state. A one-row input is the scalar deployment path.
    fn replay_batch<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<'_, i8>,
        positive_indices: PyReadonlyArray2<'_, usize>,
        negative_indices: PyReadonlyArray2<'_, usize>,
        uniforms: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyList>> {
        let states = states.as_array();
        let positive_indices = positive_indices.as_array();
        let negative_indices = negative_indices.as_array();
        let uniforms = uniforms.as_array();
        let batch = states.nrows();
        if batch == 0 {
            return Err(value_error("batch shape must contain at least one state"));
        }
        if states.ncols() != self.core.config.n_spins() {
            return Err(value_error(format!(
                "state width must be {}",
                self.core.config.n_spins()
            )));
        }
        if positive_indices.nrows() != batch
            || negative_indices.nrows() != batch
            || uniforms.nrows() != batch
        {
            return Err(value_error("batch shape must match for states and tapes"));
        }
        let tape_steps = positive_indices.ncols();
        if negative_indices.ncols() != tape_steps || uniforms.ncols() != tape_steps {
            return Err(value_error(
                "tape shape must match for indices and uniforms",
            ));
        }

        let results = PyList::empty(py);
        for batch_index in 0..batch {
            self.tape_buffer.clear();
            self.tape_buffer.reserve(tape_steps);
            for step_index in 0..tape_steps {
                self.tape_buffer.push(
                    PairSwapDraw::new(
                        positive_indices[(batch_index, step_index)],
                        negative_indices[(batch_index, step_index)],
                        uniforms[(batch_index, step_index)],
                    )
                    .map_err(value_error)?,
                );
            }
            let state = states.row(batch_index).iter().copied().collect::<Vec<_>>();
            let outcome = self
                .core
                .run_replay(&state, &self.tape_buffer)
                .map_err(value_error)?;
            results.append(replay_to_dict(py, &outcome)?)?;
        }
        self.call_count = self
            .call_count
            .checked_add(1)
            .ok_or_else(|| value_error("call count overflow"))?;
        self.max_batch_size = self.max_batch_size.max(batch);
        Ok(results)
    }

    /// Run independent Rust random streams without sharing RNG state across rows.
    fn run_seeded_batch<'py>(
        &mut self,
        py: Python<'py>,
        states: PyReadonlyArray2<'_, i8>,
        seeds: Vec<u64>,
        burn_in: usize,
        retained: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let states = states.as_array();
        if states.nrows() == 0 || states.ncols() != self.core.config.n_spins() {
            return Err(value_error(format!(
                "state batch shape must be (batch, {})",
                self.core.config.n_spins()
            )));
        }
        if seeds.len() != states.nrows() {
            return Err(value_error("seed count must match the state batch"));
        }
        let results = PyList::empty(py);
        for (batch_index, seed) in seeds.into_iter().enumerate() {
            let state = states.row(batch_index).iter().copied().collect::<Vec<_>>();
            let outcome = self
                .core
                .run_seeded(&state, seed, burn_in, retained)
                .map_err(value_error)?;
            results.append(chain_to_dict(py, &outcome)?)?;
        }
        self.call_count = self
            .call_count
            .checked_add(1)
            .ok_or_else(|| value_error("call count overflow"))?;
        self.max_batch_size = self.max_batch_size.max(states.nrows());
        Ok(results)
    }

    /// Serialize a complete restart state with the kernel's serde contract.
    fn serialize_state(&self, state: &Bound<'_, PyDict>) -> PyResult<String> {
        serde_json::to_string(&seeded_state_from_dict(state)?).map_err(value_error)
    }

    /// Load Python-provided JSON and return validated, Python-owned state fields.
    fn deserialize_state<'py>(
        &self,
        py: Python<'py>,
        serialized: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let decoded: PairSwapSeededState = serde_json::from_str(serialized).map_err(value_error)?;
        let mut validated =
            PairSwapSeededState::new(decoded.spins, decoded.rng_state).map_err(value_error)?;
        validated.transition = decoded.transition;
        seeded_state_to_dict(py, &validated)
    }

    /// Report allocation reuse without exposing process-specific pointer values.
    fn buffer_receipt<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let result = PyDict::new(py);
        result.set_item("tape_capacity", self.tape_buffer.capacity())?;
        result.set_item("call_count", self.call_count)?;
        result.set_item("max_batch_size", self.max_batch_size)?;
        Ok(result)
    }
}

pub fn register_fixed_cardinality_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "fixed_cardinality")?;
    module.add_class::<PyFixedCardinalitySampler>()?;
    parent.add_submodule(&module)?;
    parent.add_class::<PyFixedCardinalitySampler>()?;
    Ok(())
}
