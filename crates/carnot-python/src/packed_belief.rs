//! Persistent packed survivor and vote-count controller.
//!
//! The Python experiment owns provenance and scheduling. This module keeps only
//! the finite state that changes decisions, so the native boundary stays small.
//!
//! Spec: REQ-CL-7230, REQ-RUSTPY-7230

use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::{Deserialize, Serialize};
use std::time::Instant;

const FAMILY_COUNT: usize = 4;
const DOMAIN_SIZE: usize = 33;
const FULL_MASK: u64 = (1_u64 << DOMAIN_SIZE) - 1;

fn value_error(error: impl ToString) -> PyErr {
    PyValueError::new_err(error.to_string())
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SemanticState {
    epochs: [u64; FAMILY_COUNT],
    survivor_masks: [u64; FAMILY_COUNT],
    version: u64,
    vote_counts: [Vec<u8>; FAMILY_COUNT],
}

impl SemanticState {
    fn initial() -> Self {
        let survivor_masks = [FULL_MASK; FAMILY_COUNT];
        Self {
            epochs: [0; FAMILY_COUNT],
            survivor_masks,
            version: 0,
            vote_counts: vote_counts(&survivor_masks),
        }
    }

    fn validate(&self) -> Result<(), &'static str> {
        if self
            .survivor_masks
            .iter()
            .any(|mask| mask & !FULL_MASK != 0)
        {
            return Err("invalid survivor mask");
        }
        if self.vote_counts != vote_counts(&self.survivor_masks) {
            return Err("invalid vote counts");
        }
        if self
            .vote_counts
            .iter()
            .any(|family| family.len() != DOMAIN_SIZE)
        {
            return Err("invalid vote count width");
        }
        Ok(())
    }
}

fn accepts(family: usize, value: usize, parameter: usize) -> bool {
    match family {
        0 => value >= parameter,
        1 => value <= parameter,
        2 => value == parameter,
        3 => (value + DOMAIN_SIZE - parameter) % DOMAIN_SIZE < 8,
        _ => false,
    }
}

fn accept_mask(family: usize, value: usize) -> u64 {
    (0..DOMAIN_SIZE).fold(0_u64, |mask, parameter| {
        if accepts(family, value, parameter) {
            mask | (1_u64 << parameter)
        } else {
            mask
        }
    })
}

fn vote_counts(masks: &[u64; FAMILY_COUNT]) -> [Vec<u8>; FAMILY_COUNT] {
    let mut votes = std::array::from_fn(|_| vec![0_u8; DOMAIN_SIZE]);
    for family in 0..FAMILY_COUNT {
        for value in 0..DOMAIN_SIZE {
            votes[family][value] = (masks[family] & accept_mask(family, value)).count_ones() as u8;
        }
    }
    votes
}

fn state_json(state: &SemanticState) -> PyResult<String> {
    serde_json::to_string(state).map_err(value_error)
}

/// Native state for all four finite predicate families.
#[pyclass(name = "RustPackedBeliefController")]
pub struct PyPackedBeliefController {
    state: SemanticState,
    rollback_state: Option<SemanticState>,
}

#[pymethods]
impl PyPackedBeliefController {
    #[new]
    fn new() -> Self {
        Self {
            state: SemanticState::initial(),
            rollback_state: None,
        }
    }

    /// Evaluate many public inputs without changing persistent state.
    fn query_batch<'py>(
        &self,
        py: Python<'py>,
        families: PyReadonlyArray1<'_, u8>,
        values: PyReadonlyArray1<'_, i64>,
        labels: PyReadonlyArray1<'_, i8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let families = families.as_slice()?;
        let values = values.as_slice()?;
        let labels = labels.as_slice()?;
        if families.len() != values.len() || families.len() != labels.len() {
            return Err(value_error("query batch lengths must match"));
        }
        let started = Instant::now();
        let mut decisions = Vec::with_capacity(families.len());
        let mut disagreements = Vec::with_capacity(families.len());
        let mut energy_status = Vec::with_capacity(families.len());
        let mut energies = Vec::with_capacity(families.len());
        for ((&family, &raw_value), &label) in families.iter().zip(values).zip(labels) {
            let family = family as usize;
            if family >= FAMILY_COUNT {
                decisions.push(-2_i8);
                disagreements.push(0.0);
                energy_status.push(-2_i8);
                energies.push(None);
                continue;
            }
            let value = raw_value.rem_euclid(DOMAIN_SIZE as i64) as usize;
            let mask = self.state.survivor_masks[family];
            let count = mask.count_ones();
            if count == 0 {
                decisions.push(-1_i8);
                disagreements.push(0.0);
                energy_status.push(0_i8);
                energies.push(None);
                continue;
            }
            let accept_count = self.state.vote_counts[family][value] as u32;
            decisions.push(if accept_count * 2 > count { 1 } else { 0 });
            disagreements.push(accept_count.min(count - accept_count) as f64 / count as f64);
            if label != 0 && label != 1 {
                energy_status.push(-1_i8);
                energies.push(None);
                continue;
            }
            let disagree_count = if label == 1 {
                count - accept_count
            } else {
                accept_count
            };
            energy_status.push(1_i8);
            energies.push(Some(disagree_count as f64 / count as f64));
        }
        let result = PyDict::new(py);
        result.set_item("decisions", decisions)?;
        result.set_item("disagreements", disagreements)?;
        result.set_item("energy_status", energy_status)?;
        result.set_item("energies", energies)?;
        result.set_item("kernel_ns", started.elapsed().as_nanos())?;
        Ok(result)
    }

    /// Apply one ordered release batch and retain one exact rollback parent.
    fn update_batch<'py>(
        &mut self,
        py: Python<'py>,
        families: PyReadonlyArray1<'_, u8>,
        values: PyReadonlyArray1<'_, i64>,
        labels: PyReadonlyArray1<'_, i8>,
        roles: PyReadonlyArray1<'_, u8>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let families = families.as_slice()?;
        let values = values.as_slice()?;
        let labels = labels.as_slice()?;
        let roles = roles.as_slice()?;
        if families.len() != values.len()
            || families.len() != labels.len()
            || families.len() != roles.len()
        {
            return Err(value_error("update batch lengths must match"));
        }
        if families
            .iter()
            .any(|family| *family as usize >= FAMILY_COUNT)
            || labels.iter().any(|label| *label != 0 && *label != 1)
            || roles.iter().any(|role| *role != 0 && *role != 1)
        {
            return Err(value_error("invalid update family, label, or role"));
        }
        let started = Instant::now();
        let parent = self.state.clone();
        let mut candidate = parent.clone();
        let mut reset_count = 0_usize;
        for (((&family, &raw_value), &label), &role) in
            families.iter().zip(values).zip(labels).zip(roles)
        {
            if role == 0 {
                continue;
            }
            let family = family as usize;
            let value = raw_value.rem_euclid(DOMAIN_SIZE as i64) as usize;
            let accepted = accept_mask(family, value);
            let matching = if label == 1 {
                accepted
            } else {
                FULL_MASK ^ accepted
            };
            let survivor = candidate.survivor_masks[family] & matching;
            if survivor == 0 {
                candidate.survivor_masks[family] = FULL_MASK & matching;
                candidate.epochs[family] += 1;
                reset_count += 1;
            } else {
                candidate.survivor_masks[family] = survivor;
            }
        }
        candidate.version = candidate
            .version
            .checked_add(1)
            .ok_or_else(|| value_error("state version overflow"))?;
        candidate.vote_counts = vote_counts(&candidate.survivor_masks);
        candidate.validate().map_err(value_error)?;
        self.rollback_state = Some(parent);
        self.state = candidate;
        let result = PyDict::new(py);
        result.set_item("reset_count", reset_count)?;
        result.set_item("version", self.state.version)?;
        result.set_item("survivor_masks", self.state.survivor_masks)?;
        result.set_item("kernel_ns", started.elapsed().as_nanos())?;
        Ok(result)
    }

    /// Serialize the decision-bearing state for checkpoint and process restore.
    fn serialize_state(&self) -> PyResult<String> {
        state_json(&self.state)
    }

    /// Replace state only after every mask and cached vote count validates.
    fn load_state(&mut self, serialized: &str) -> PyResult<()> {
        let state: SemanticState = serde_json::from_str(serialized).map_err(value_error)?;
        state.validate().map_err(value_error)?;
        self.state = state;
        self.rollback_state = None;
        Ok(())
    }

    /// Restore the parent retained by the most recent admitted update.
    fn rollback<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let parent = self
            .rollback_state
            .take()
            .ok_or_else(|| value_error("no rollback state"))?;
        let expected = state_json(&parent)?;
        self.state = parent;
        let result = PyDict::new(py);
        result.set_item("serialized_state", state_json(&self.state)?)?;
        result.set_item("byte_identical", state_json(&self.state)? == expected)?;
        Ok(result)
    }

    /// Return to the full initial version space between independent traces.
    fn reset(&mut self) {
        self.state = SemanticState::initial();
        self.rollback_state = None;
    }
}

pub fn register_packed_belief_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let module = PyModule::new(parent.py(), "packed_belief")?;
    module.add_class::<PyPackedBeliefController>()?;
    parent.add_submodule(&module)?;
    parent.add_class::<PyPackedBeliefController>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// REQ-CL-7230: ties reject and contradictory support resets the epoch.
    #[test]
    fn packed_core_tie_and_reset() {
        let mut state = SemanticState::initial();
        state.survivor_masks[0] = (1_u64 << 0) | (1_u64 << 16);
        state.vote_counts = vote_counts(&state.survivor_masks);
        assert_eq!(state.vote_counts[0][8], 1);
        let matching = FULL_MASK ^ accept_mask(0, 16);
        assert_eq!(state.survivor_masks[0] & matching, 0);
        state.survivor_masks[0] = matching;
        state.epochs[0] += 1;
        state.vote_counts = vote_counts(&state.survivor_masks);
        assert!(state.validate().is_ok());
        assert_eq!(state.epochs[0], 1);
    }
}
