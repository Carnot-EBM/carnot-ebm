//! Fixed-cardinality pair-swap Metropolis sampling for finite Ising models.
//!
//! The caller can supply every proposal and acceptance draw. This makes an
//! exact cross-language replay possible without assuming that two random
//! number generators produce the same stream.
//!
//! Spec: REQ-SAMPLER-7189, SCENARIO-SAMPLER-7189-REPLAY

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;

/// One undirected Ising edge. Each pair is stored once with `left < right`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredEdge {
    pub left: usize,
    pub right: usize,
    pub coupling: f64,
}

impl StoredEdge {
    pub fn new(left: usize, right: usize, coupling: f64) -> Self {
        Self {
            left,
            right,
            coupling,
        }
    }

    pub fn try_new(left: usize, right: usize, coupling: f64) -> Result<Self, String> {
        let edge = Self::new(left, right, coupling);
        edge.validate()?;
        Ok(edge)
    }

    fn validate(&self) -> Result<(), String> {
        if self.left >= self.right {
            return Err("stored edges must satisfy left < right".to_string());
        }
        if !self.coupling.is_finite() {
            return Err("couplings must be finite".to_string());
        }
        Ok(())
    }
}

/// Immutable model and slice parameters for the pair-swap kernel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSwapConfig {
    pub edges: Vec<StoredEdge>,
    pub fields: Vec<f64>,
    pub cardinality: usize,
    pub beta: f64,
}

impl PairSwapConfig {
    pub fn new(
        edges: Vec<StoredEdge>,
        fields: Vec<f64>,
        cardinality: usize,
        beta: f64,
    ) -> Result<Self, String> {
        if fields.is_empty() {
            return Err("fields must be non-empty".to_string());
        }
        if fields.iter().any(|value| !value.is_finite() || *value == 0.0) {
            return Err("fields must be finite and nonzero".to_string());
        }
        if cardinality > fields.len() {
            return Err("cardinality must not exceed the state length".to_string());
        }
        if !beta.is_finite() || beta <= 0.0 {
            return Err("beta must be finite and positive".to_string());
        }
        let mut pairs = HashSet::new();
        for edge in &edges {
            edge.validate()?;
            if edge.right >= fields.len() {
                return Err("edge endpoint is outside the state".to_string());
            }
            if !pairs.insert((edge.left, edge.right)) {
                return Err("stored edge pairs must be unique".to_string());
            }
        }
        Ok(Self {
            edges,
            fields,
            cardinality,
            beta,
        })
    }

    pub fn n_spins(&self) -> usize {
        self.fields.len()
    }
}

/// One deterministic proposal-index and acceptance-uniform tape entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSwapDraw {
    pub positive_index: usize,
    pub negative_index: usize,
    pub uniform: f64,
}

impl PairSwapDraw {
    pub fn new(
        positive_index: usize,
        negative_index: usize,
        uniform: f64,
    ) -> Result<Self, String> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err("uniform must be finite and in [0, 1)".to_string());
        }
        Ok(Self {
            positive_index,
            negative_index,
            uniform,
        })
    }
}

/// Full diagnostic for one deterministic pair-swap transition.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSwapStepOutcome {
    pub state: Vec<i8>,
    pub proposed_state: Vec<i8>,
    pub current_energy: f64,
    pub proposed_energy: f64,
    pub delta_energy: f64,
    pub log_acceptance: f64,
    pub accepted: bool,
    pub cardinality: usize,
}

/// Full output from replaying a caller-owned random tape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSwapReplayOutcome {
    pub initial_state: Vec<i8>,
    pub final_state: Vec<i8>,
    pub steps: Vec<PairSwapStepOutcome>,
}

/// Restartable state for an independent Rust random stream.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PairSwapSeededState {
    pub spins: Vec<i8>,
    pub rng_state: u64,
    pub transition: usize,
}

impl PairSwapSeededState {
    pub fn new(spins: Vec<i8>, seed: u64) -> Result<Self, String> {
        validate_spin_values(&spins)?;
        Ok(Self {
            spins,
            rng_state: seed,
            transition: 0,
        })
    }
}

/// Samples and energies from a fixed-length independent Rust stream.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PairSwapChainOutcome {
    pub samples: Vec<Vec<i8>>,
    pub energies: Vec<f64>,
    pub accepted: usize,
    pub attempted: usize,
    pub final_state: PairSwapSeededState,
}

/// Exact energy and transition core for one fixed-cardinality Ising slice.
#[derive(Clone, Debug, PartialEq)]
pub struct PairSwapCore {
    pub config: PairSwapConfig,
}

impl PairSwapCore {
    pub fn new(config: PairSwapConfig) -> Self {
        Self { config }
    }

    /// Compute the once-stored-edge convention used by Exp7187.
    pub fn energy(&self, state: &[i8]) -> Result<f64, String> {
        self.validate_state(state)?;
        let edge_term: f64 = self
            .config
            .edges
            .iter()
            .map(|edge| {
                edge.coupling
                    * f64::from(state[edge.left])
                    * f64::from(state[edge.right])
            })
            .sum();
        let field_term: f64 = self
            .config
            .fields
            .iter()
            .zip(state.iter())
            .map(|(field, spin)| field * f64::from(*spin))
            .sum();
        Ok(-edge_term - field_term)
    }

    /// Apply one transition from explicit proposal indices and a uniform draw.
    pub fn step_from_draw(
        &self,
        state: &[i8],
        draw: &PairSwapDraw,
    ) -> Result<PairSwapStepOutcome, String> {
        self.validate_state(state)?;
        if !draw.uniform.is_finite() || !(0.0..1.0).contains(&draw.uniform) {
            return Err("uniform must be finite and in [0, 1)".to_string());
        }
        let current_energy = self.energy(state)?;
        if self.config.cardinality == 0 || self.config.cardinality == self.config.n_spins() {
            return Ok(PairSwapStepOutcome {
                state: state.to_vec(),
                proposed_state: state.to_vec(),
                current_energy,
                proposed_energy: current_energy,
                delta_energy: 0.0,
                log_acceptance: 0.0,
                accepted: true,
                cardinality: self.config.cardinality,
            });
        }
        let positive: Vec<usize> = state
            .iter()
            .enumerate()
            .filter_map(|(index, spin)| (*spin == 1).then_some(index))
            .collect();
        let negative: Vec<usize> = state
            .iter()
            .enumerate()
            .filter_map(|(index, spin)| (*spin == -1).then_some(index))
            .collect();
        let positive_site = *positive
            .get(draw.positive_index)
            .ok_or_else(|| "positive proposal index is out of range".to_string())?;
        let negative_site = *negative
            .get(draw.negative_index)
            .ok_or_else(|| "negative proposal index is out of range".to_string())?;
        let mut proposed_state = state.to_vec();
        proposed_state.swap(positive_site, negative_site);
        let proposed_energy = self.energy(&proposed_state)?;
        let delta_energy = proposed_energy - current_energy;
        let log_acceptance = (-self.config.beta * delta_energy).min(0.0);
        let accepted = draw.uniform.max(f64::MIN_POSITIVE).ln() < log_acceptance;
        let next_state = if accepted {
            proposed_state.clone()
        } else {
            state.to_vec()
        };
        Ok(PairSwapStepOutcome {
            state: next_state,
            proposed_state,
            current_energy,
            proposed_energy,
            delta_energy,
            log_acceptance,
            accepted,
            cardinality: self.config.cardinality,
        })
    }

    /// Replay every caller-owned draw and retain each transition diagnostic.
    pub fn run_replay(
        &self,
        initial_state: &[i8],
        tape: &[PairSwapDraw],
    ) -> Result<PairSwapReplayOutcome, String> {
        self.validate_state(initial_state)?;
        let mut state = initial_state.to_vec();
        let mut steps = Vec::with_capacity(tape.len());
        for draw in tape {
            let outcome = self.step_from_draw(&state, draw)?;
            state = outcome.state.clone();
            steps.push(outcome);
        }
        Ok(PairSwapReplayOutcome {
            initial_state: initial_state.to_vec(),
            final_state: state,
            steps,
        })
    }

    /// Advance one independent Rust stream with three internal uniform draws.
    pub fn step_seeded(
        &self,
        state: &mut PairSwapSeededState,
    ) -> Result<PairSwapStepOutcome, String> {
        self.validate_state(&state.spins)?;
        let positive_count = self.config.cardinality.max(1);
        let negative_count = (self.config.n_spins() - self.config.cardinality).max(1);
        let positive_index = uniform_index(next_uniform(&mut state.rng_state), positive_count);
        let negative_index = uniform_index(next_uniform(&mut state.rng_state), negative_count);
        let uniform = next_uniform(&mut state.rng_state);
        let draw = PairSwapDraw::new(positive_index, negative_index, uniform)?;
        let outcome = self.step_from_draw(&state.spins, &draw)?;
        state.spins = outcome.state.clone();
        state.transition = state
            .transition
            .checked_add(1)
            .ok_or_else(|| "transition count overflow".to_string())?;
        Ok(outcome)
    }

    /// Run burn-in and retain the requested number of samples and energies.
    pub fn run_seeded(
        &self,
        initial_state: &[i8],
        seed: u64,
        burn_in: usize,
        retained: usize,
    ) -> Result<PairSwapChainOutcome, String> {
        if retained == 0 {
            return Err("retained sample count must be positive".to_string());
        }
        let total = burn_in
            .checked_add(retained)
            .ok_or_else(|| "transition count overflow".to_string())?;
        let mut state = PairSwapSeededState::new(initial_state.to_vec(), seed)?;
        self.validate_state(&state.spins)?;
        let mut samples = Vec::with_capacity(retained);
        let mut energies = Vec::with_capacity(retained);
        let mut accepted = 0;
        for index in 0..total {
            let outcome = self.step_seeded(&mut state)?;
            accepted += usize::from(outcome.accepted);
            if index >= burn_in {
                energies.push(self.energy(&state.spins)?);
                samples.push(state.spins.clone());
            }
        }
        Ok(PairSwapChainOutcome {
            samples,
            energies,
            accepted,
            attempted: total,
            final_state: state,
        })
    }

    fn validate_state(&self, state: &[i8]) -> Result<(), String> {
        validate_spin_values(state)?;
        if state.len() != self.config.n_spins() {
            return Err(format!(
                "state length must be {}",
                self.config.n_spins()
            ));
        }
        let cardinality = state.iter().filter(|spin| **spin == 1).count();
        if cardinality != self.config.cardinality {
            return Err(format!(
                "state cardinality must be {}, got {cardinality}",
                self.config.cardinality
            ));
        }
        Ok(())
    }
}

fn validate_spin_values(state: &[i8]) -> Result<(), String> {
    if state.is_empty() || state.iter().any(|spin| !matches!(spin, -1 | 1)) {
        return Err("state must contain only -1 and +1 spins".to_string());
    }
    Ok(())
}

fn next_uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    ((*state >> 11) as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn uniform_index(uniform: f64, count: usize) -> usize {
    ((uniform * count as f64) as usize).min(count - 1)
}
