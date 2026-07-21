//! One-axis corrected-cDLS temperature-label replica exchange.
//!
//! This module is intentionally small: it implements only the promoted
//! one-axis method from Exp5633/5634. It does not implement penalty-axis
//! exchange. The public API is deterministic so the PyO3 wrapper can replay
//! exact Python/Rust parity checks and checkpoint restarts.
//!
//! Spec: REQ-SAMPLE-5714, SCENARIO-SAMPLE-5714

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

/// Frozen one-axis corrected-cDLS configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisTemperingConfig {
    pub couplings: Vec<Vec<f64>>,
    pub fields: Vec<f64>,
    pub beta_ladder: Vec<f64>,
    pub proposal_std: f64,
    pub drift_scale: f64,
}

impl OneAxisTemperingConfig {
    /// Validate and build the finite Ising/temperature-ladder configuration.
    pub fn new(
        couplings: Vec<Vec<f64>>,
        fields: Vec<f64>,
        beta_ladder: Vec<f64>,
        proposal_std: f64,
        drift_scale: f64,
    ) -> Result<Self, String> {
        if couplings.is_empty() {
            return Err("couplings must be square and non-empty".to_string());
        }
        let n = couplings.len();
        if couplings.iter().any(|row| row.len() != n) {
            return Err("couplings must be square".to_string());
        }
        if fields.len() != n {
            return Err(format!(
                "fields length must match couplings dimension: expected {n}, got {}",
                fields.len()
            ));
        }
        if couplings
            .iter()
            .flatten()
            .chain(fields.iter())
            .any(|value| !value.is_finite())
        {
            return Err("couplings and fields must be finite".to_string());
        }
        if beta_ladder.len() < 2 {
            return Err("beta_ladder must contain at least two labels".to_string());
        }
        if beta_ladder
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err("beta_ladder values must be finite and positive".to_string());
        }
        for pair in beta_ladder.windows(2) {
            if pair[0] >= pair[1] {
                return Err("beta_ladder must be strictly increasing".to_string());
            }
        }
        if !proposal_std.is_finite() || proposal_std <= 0.0 {
            return Err("proposal_std must be finite and positive".to_string());
        }
        if !drift_scale.is_finite() {
            return Err("drift_scale must be finite".to_string());
        }
        Ok(Self {
            couplings,
            fields,
            beta_ladder,
            proposal_std,
            drift_scale,
        })
    }

    pub fn n_spins(&self) -> usize {
        self.fields.len()
    }
}

/// Serializable seeded state for one-axis replica exchange.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneAxisTemperingState {
    pub states: Vec<Vec<i8>>,
    pub labels: Vec<usize>,
    pub rng_state: u64,
    pub sweep: usize,
}

impl OneAxisTemperingState {
    /// Validate labels as a permutation and states as Ising spins.
    pub fn new(
        states: Vec<Vec<i8>>,
        labels: Vec<usize>,
        rng_state: u64,
        sweep: usize,
    ) -> Result<Self, String> {
        if states.is_empty() || states.len() != labels.len() {
            return Err("states and labels must have the same non-empty replica count".to_string());
        }
        let n_spins = states[0].len();
        if n_spins == 0 {
            return Err("states must contain at least one spin".to_string());
        }
        for state in &states {
            validate_spin_state(state, n_spins)?;
        }
        validate_labels(&labels)?;
        Ok(Self {
            states,
            labels,
            rng_state,
            sweep,
        })
    }
}

/// Deterministic corrected within-replica transition diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct CorrectedStepOutcome {
    pub state: Vec<i8>,
    pub proposed_state: Vec<i8>,
    pub current_energy: f64,
    pub proposed_energy: f64,
    pub proposal_log_forward: f64,
    pub proposal_log_reverse: f64,
    pub log_acceptance: f64,
    pub accepted: bool,
}

/// Deterministic temperature-label swap diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct SwapOutcome {
    pub labels: Vec<usize>,
    pub proposed_labels: Vec<usize>,
    pub log_ratio: f64,
    pub acceptance_probability: f64,
    pub accepted: bool,
}

/// Allocation counters for the compact production hot path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactRunCounters {
    pub rust_per_sample_heap_allocations: usize,
    pub workspace_allocations: usize,
    pub output_allocations: usize,
    pub total_corrected_transitions: usize,
    pub total_swap_attempts: usize,
}

/// Buffer-reuse receipt for the compact production hot path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactBufferReuseReceipt {
    pub contiguous_samples: bool,
    pub workspace_reused: bool,
    pub per_sample_heap_buffers: usize,
}

/// Worker policy receipt for the compact production hot path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactWorkerPoolReceipt {
    pub fixed_worker_count: usize,
    pub dynamic_per_sample_workers: usize,
}

/// Compact sweep output used when Python does not need per-transition diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct CompactRunOutcome {
    pub samples_spin: Vec<i8>,
    pub final_state: OneAxisTemperingState,
    pub counters: CompactRunCounters,
    pub buffer_reuse: CompactBufferReuseReceipt,
    pub worker_pool: CompactWorkerPoolReceipt,
}

/// Exact one-axis energy/scheduler core.
#[derive(Clone, Debug, PartialEq)]
pub struct OneAxisTemperingCore {
    pub config: OneAxisTemperingConfig,
}

impl OneAxisTemperingCore {
    pub fn new(config: OneAxisTemperingConfig) -> Self {
        Self { config }
    }

    /// Compute `E(x) = -0.5 x^T J x - h^T x` for an Ising state.
    pub fn energy(&self, state: &[i8]) -> Result<f64, String> {
        validate_spin_state(state, self.config.n_spins())?;
        let spin_values: Vec<f64> = state.iter().map(|value| f64::from(*value)).collect();
        let mut projected = vec![0.0; self.config.n_spins()];
        for (i, spin) in spin_values.iter().enumerate() {
            for (j, coupling) in self.config.couplings[i].iter().enumerate() {
                projected[j] += spin * coupling;
            }
        }
        let pair_term: f64 = projected
            .iter()
            .zip(spin_values.iter())
            .map(|(projected_value, spin)| projected_value * spin)
            .sum();
        let field_term: f64 = spin_values
            .iter()
            .zip(self.config.fields.iter())
            .map(|(spin, field)| spin * field)
            .sum();
        Ok(-0.5 * pair_term - field_term)
    }

    /// Return log q(target | source) for Exp5622's projected Gaussian proposal.
    pub fn proposal_log_probability(
        &self,
        source: &[i8],
        target: &[i8],
        beta: f64,
    ) -> Result<f64, String> {
        validate_beta(beta)?;
        validate_spin_state(source, self.config.n_spins())?;
        validate_spin_state(target, self.config.n_spins())?;
        let mean = self.proposal_mean(source, beta)?;
        let mut log_probability = 0.0;
        for (sign, coordinate_mean) in target.iter().zip(mean.iter()) {
            let probability =
                normal_cdf(f64::from(*sign) * *coordinate_mean / self.config.proposal_std);
            log_probability += probability.ln();
        }
        Ok(log_probability)
    }

    /// Run one corrected cDLS proposal from caller-provided uniforms.
    pub fn corrected_step(
        &self,
        state: &[i8],
        beta: f64,
        uniforms: &[f64],
    ) -> Result<CorrectedStepOutcome, String> {
        validate_beta(beta)?;
        validate_spin_state(state, self.config.n_spins())?;
        if uniforms.len() != self.config.n_spins() + 1 {
            return Err(format!(
                "uniforms length must be n_spins + 1: expected {}, got {}",
                self.config.n_spins() + 1,
                uniforms.len()
            ));
        }
        for value in uniforms {
            if !value.is_finite() || *value < 0.0 || *value >= 1.0 {
                return Err("uniforms must be finite values in [0, 1)".to_string());
            }
        }

        let proposed_state = self.projected_proposal_from_uniforms(state, beta, uniforms)?;
        let current_energy = self.energy(state)?;
        let proposed_energy = self.energy(&proposed_state)?;
        let proposal_log_forward = self.proposal_log_probability(state, &proposed_state, beta)?;
        let proposal_log_reverse = self.proposal_log_probability(&proposed_state, state, beta)?;
        let log_acceptance = -beta * (proposed_energy - current_energy) + proposal_log_reverse
            - proposal_log_forward;
        let accept_uniform = uniforms[self.config.n_spins()];
        let accepted = log_acceptance >= 0.0 || accept_uniform.ln() < log_acceptance;
        Ok(CorrectedStepOutcome {
            state: if accepted {
                proposed_state.clone()
            } else {
                state.to_vec()
            },
            proposed_state,
            current_energy,
            proposed_energy,
            proposal_log_forward,
            proposal_log_reverse,
            log_acceptance,
            accepted,
        })
    }

    /// Compute the exact Exp5633 adjacent temperature-label log ratio.
    pub fn swap_log_ratio(
        &self,
        states: &[Vec<i8>],
        labels: &[usize],
        label_pair: &[usize],
    ) -> Result<f64, String> {
        validate_state_collection(
            states,
            labels,
            self.config.n_spins(),
            self.config.beta_ladder.len(),
        )?;
        let pair = validate_label_pair(label_pair, self.config.beta_ladder.len())?;
        let left_pos = label_position(labels, pair.0)?;
        let right_pos = label_position(labels, pair.1)?;
        let beta_left = self.config.beta_ladder[pair.0];
        let beta_right = self.config.beta_ladder[pair.1];
        let energy_left = self.energy(&states[left_pos])?;
        let energy_right = self.energy(&states[right_pos])?;
        Ok((beta_left - beta_right) * (energy_left - energy_right))
    }

    /// Decide an adjacent temperature-label swap from a caller-provided uniform.
    pub fn swap_decision(
        &self,
        states: &[Vec<i8>],
        labels: &[usize],
        label_pair: &[usize],
        uniform: f64,
    ) -> Result<SwapOutcome, String> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err("swap uniform must be finite and in [0, 1)".to_string());
        }
        let pair = validate_label_pair(label_pair, self.config.beta_ladder.len())?;
        let log_ratio = self.swap_log_ratio(states, labels, label_pair)?;
        let acceptance_probability = acceptance_probability(log_ratio);
        let accepted = uniform < acceptance_probability;
        let proposed_labels = swap_labels(labels, pair)?;
        Ok(SwapOutcome {
            labels: if accepted {
                proposed_labels.clone()
            } else {
                labels.to_vec()
            },
            proposed_labels,
            log_ratio,
            acceptance_probability,
            accepted,
        })
    }

    /// Return the fixed one-axis schedule as compact stable labels.
    pub fn scheduler_trace(&self) -> Vec<String> {
        let mut steps = Vec::with_capacity(self.config.beta_ladder.len() * 2 - 1);
        for replica in 0..self.config.beta_ladder.len() {
            steps.push(format!("within:{replica}"));
        }
        for left in 0..(self.config.beta_ladder.len() - 1) {
            steps.push(format!("swap:{left}-{}", left + 1));
        }
        steps
    }

    /// Run one full within-replica plus adjacent-label exchange sweep.
    pub fn step(&self, state: &OneAxisTemperingState) -> Result<OneAxisTemperingState, String> {
        validate_state_collection(
            &state.states,
            &state.labels,
            self.config.n_spins(),
            self.config.beta_ladder.len(),
        )?;
        let mut next = state.clone();
        for physical_index in 0..self.config.beta_ladder.len() {
            let beta_label = next.labels[physical_index];
            let beta = self.config.beta_ladder[beta_label];
            let uniforms = draw_uniforms(&mut next.rng_state, self.config.n_spins() + 1);
            let outcome = self.corrected_step(&next.states[physical_index], beta, &uniforms)?;
            next.states[physical_index] = outcome.state;
        }
        for left in 0..(self.config.beta_ladder.len() - 1) {
            let uniform = next_uniform(&mut next.rng_state);
            let outcome =
                self.swap_decision(&next.states, &next.labels, &[left, left + 1], uniform)?;
            next.labels = outcome.labels;
        }
        next.sweep += 1;
        Ok(next)
    }

    /// Extract the state currently wearing the coldest/highest beta label.
    pub fn target_state(&self, state: &OneAxisTemperingState) -> Result<Vec<i8>, String> {
        validate_state_collection(
            &state.states,
            &state.labels,
            self.config.n_spins(),
            self.config.beta_ladder.len(),
        )?;
        let cold_label = self.config.beta_ladder.len() - 1;
        let position = label_position(&state.labels, cold_label)?;
        Ok(state.states[position].clone())
    }

    /// Run sweeps with compact sample output and reusable work buffers.
    pub fn run_compact_sweeps(
        &self,
        state: &OneAxisTemperingState,
        burn_in_sweeps: usize,
        n_samples: usize,
    ) -> Result<CompactRunOutcome, String> {
        if n_samples == 0 {
            return Err("n_samples must be positive".to_string());
        }
        let total_sweeps = burn_in_sweeps
            .checked_add(n_samples)
            .ok_or_else(|| "sweep count overflow".to_string())?;
        let replica_count = self.config.beta_ladder.len();
        let n_spins = self.config.n_spins();
        validate_state_collection(&state.states, &state.labels, n_spins, replica_count)?;

        let mut next = state.clone();
        let mut samples_spin = Vec::with_capacity(n_samples * n_spins);
        let mut uniforms = vec![0.0; n_spins + 1];
        let mut proposed_state = vec![1_i8; n_spins];
        let mut forward_mean = vec![0.0; n_spins];
        let mut reverse_mean = vec![0.0; n_spins];

        for local_sweep in 0..total_sweeps {
            let completed_sweep = next.sweep + 1;
            for physical_index in 0..replica_count {
                let beta_label = next.labels[physical_index];
                let beta = self.config.beta_ladder[beta_label];
                draw_uniforms_into(&mut next.rng_state, &mut uniforms);
                self.corrected_step_in_place(
                    &mut next.states[physical_index],
                    beta,
                    &uniforms,
                    &mut proposed_state,
                    &mut forward_mean,
                    &mut reverse_mean,
                )?;
            }

            for left in 0..(replica_count - 1) {
                let uniform = next_uniform(&mut next.rng_state);
                self.swap_adjacent_labels_in_place(&next.states, &mut next.labels, left, uniform)?;
            }

            next.sweep = completed_sweep;
            if local_sweep >= burn_in_sweeps {
                let cold_label = replica_count - 1;
                let position = label_position(&next.labels, cold_label)?;
                samples_spin.extend_from_slice(&next.states[position]);
            }
        }

        Ok(CompactRunOutcome {
            samples_spin,
            final_state: next,
            counters: CompactRunCounters {
                rust_per_sample_heap_allocations: 0,
                workspace_allocations: 4,
                output_allocations: 1,
                total_corrected_transitions: total_sweeps * replica_count,
                total_swap_attempts: total_sweeps * (replica_count - 1),
            },
            buffer_reuse: CompactBufferReuseReceipt {
                contiguous_samples: true,
                workspace_reused: true,
                per_sample_heap_buffers: 0,
            },
            worker_pool: CompactWorkerPoolReceipt {
                fixed_worker_count: 1,
                dynamic_per_sample_workers: 0,
            },
        })
    }

    fn proposal_mean(&self, source: &[i8], beta: f64) -> Result<Vec<f64>, String> {
        validate_spin_state(source, self.config.n_spins())?;
        let spin_values: Vec<f64> = source.iter().map(|value| f64::from(*value)).collect();
        let mut output = Vec::with_capacity(self.config.n_spins());
        for i in 0..self.config.n_spins() {
            let field = self.config.couplings[i]
                .iter()
                .zip(spin_values.iter())
                .map(|(coupling, spin)| coupling * spin)
                .sum::<f64>()
                + self.config.fields[i];
            output.push(spin_values[i] + self.config.drift_scale * beta * field);
        }
        Ok(output)
    }

    fn projected_proposal_from_uniforms(
        &self,
        source: &[i8],
        beta: f64,
        uniforms: &[f64],
    ) -> Result<Vec<i8>, String> {
        let mean = self.proposal_mean(source, beta)?;
        Ok(mean
            .iter()
            .zip(uniforms.iter())
            .take(self.config.n_spins())
            .map(|(coordinate_mean, uniform)| {
                let probability_plus = normal_cdf(*coordinate_mean / self.config.proposal_std);
                if *uniform < probability_plus {
                    1
                } else {
                    -1
                }
            })
            .collect())
    }

    fn corrected_step_in_place(
        &self,
        state: &mut [i8],
        beta: f64,
        uniforms: &[f64],
        proposed_state: &mut [i8],
        forward_mean: &mut [f64],
        reverse_mean: &mut [f64],
    ) -> Result<(), String> {
        validate_beta(beta)?;
        validate_spin_state(state, self.config.n_spins())?;
        if uniforms.len() != self.config.n_spins() + 1 {
            return Err(format!(
                "uniforms length must be n_spins + 1: expected {}, got {}",
                self.config.n_spins() + 1,
                uniforms.len()
            ));
        }
        if proposed_state.len() != self.config.n_spins()
            || forward_mean.len() != self.config.n_spins()
            || reverse_mean.len() != self.config.n_spins()
        {
            return Err("compact work buffers must match n_spins".to_string());
        }
        for value in uniforms {
            if !value.is_finite() || *value < 0.0 || *value >= 1.0 {
                return Err("uniforms must be finite values in [0, 1)".to_string());
            }
        }

        self.proposal_mean_into(state, beta, forward_mean);
        for i in 0..self.config.n_spins() {
            let probability_plus = normal_cdf(forward_mean[i] / self.config.proposal_std);
            proposed_state[i] = if uniforms[i] < probability_plus {
                1
            } else {
                -1
            };
        }
        let current_energy = self.energy_no_alloc(state);
        let proposed_energy = self.energy_no_alloc(proposed_state);
        let proposal_log_forward =
            self.proposal_log_probability_with_mean(proposed_state, forward_mean);
        self.proposal_mean_into(proposed_state, beta, reverse_mean);
        let proposal_log_reverse = self.proposal_log_probability_with_mean(state, reverse_mean);
        let log_acceptance = -beta * (proposed_energy - current_energy) + proposal_log_reverse
            - proposal_log_forward;
        let accept_uniform = uniforms[self.config.n_spins()];
        let accepted = log_acceptance >= 0.0 || accept_uniform.ln() < log_acceptance;
        if accepted {
            state.copy_from_slice(proposed_state);
        }
        Ok(())
    }

    fn proposal_mean_into(&self, source: &[i8], beta: f64, output: &mut [f64]) {
        for i in 0..self.config.n_spins() {
            let mut field = self.config.fields[i];
            for (coupling, spin) in self.config.couplings[i].iter().zip(source.iter()) {
                field += coupling * f64::from(*spin);
            }
            output[i] = f64::from(source[i]) + self.config.drift_scale * beta * field;
        }
    }

    fn proposal_log_probability_with_mean(&self, target: &[i8], mean: &[f64]) -> f64 {
        let mut log_probability = 0.0;
        for (sign, coordinate_mean) in target.iter().zip(mean.iter()) {
            let probability =
                normal_cdf(f64::from(*sign) * *coordinate_mean / self.config.proposal_std);
            log_probability += probability.ln();
        }
        log_probability
    }

    fn energy_no_alloc(&self, state: &[i8]) -> f64 {
        let n_spins = self.config.n_spins();
        let mut pair_term = 0.0;
        for i in 0..n_spins {
            let spin_i = f64::from(state[i]);
            for j in 0..n_spins {
                pair_term += spin_i * self.config.couplings[i][j] * f64::from(state[j]);
            }
        }
        let mut field_term = 0.0;
        for (spin, field) in state.iter().zip(self.config.fields.iter()) {
            field_term += f64::from(*spin) * field;
        }
        -0.5 * pair_term - field_term
    }

    fn swap_adjacent_labels_in_place(
        &self,
        states: &[Vec<i8>],
        labels: &mut [usize],
        left: usize,
        uniform: f64,
    ) -> Result<(), String> {
        if !uniform.is_finite() || !(0.0..1.0).contains(&uniform) {
            return Err("swap uniform must be finite and in [0, 1)".to_string());
        }
        let right = left + 1;
        if right >= self.config.beta_ladder.len() {
            return Err("label_pair must contain adjacent beta-label indices".to_string());
        }
        let left_pos = label_position(labels, left)?;
        let right_pos = label_position(labels, right)?;
        let beta_left = self.config.beta_ladder[left];
        let beta_right = self.config.beta_ladder[right];
        let energy_left = self.energy_no_alloc(&states[left_pos]);
        let energy_right = self.energy_no_alloc(&states[right_pos]);
        let log_ratio = (beta_left - beta_right) * (energy_left - energy_right);
        if uniform < acceptance_probability(log_ratio) {
            labels.swap(left_pos, right_pos);
        }
        Ok(())
    }
}

fn validate_spin_state(state: &[i8], expected: usize) -> Result<(), String> {
    if state.len() != expected {
        return Err(format!(
            "state dimension mismatch: expected {expected}, got {}",
            state.len()
        ));
    }
    if state.iter().any(|value| *value != -1 && *value != 1) {
        return Err("spin state values must be -1 or +1".to_string());
    }
    Ok(())
}

fn validate_beta(beta: f64) -> Result<(), String> {
    if !beta.is_finite() || beta <= 0.0 {
        return Err("beta must be finite and positive".to_string());
    }
    Ok(())
}

fn validate_labels(labels: &[usize]) -> Result<(), String> {
    let replica_count = labels.len();
    let mut seen = vec![false; replica_count];
    for label in labels {
        if *label >= replica_count || seen[*label] {
            return Err("labels must be a permutation of beta-label indices".to_string());
        }
        seen[*label] = true;
    }
    Ok(())
}

fn validate_state_collection(
    states: &[Vec<i8>],
    labels: &[usize],
    n_spins: usize,
    replica_count: usize,
) -> Result<(), String> {
    if states.len() != replica_count || labels.len() != replica_count {
        return Err(format!(
            "states and labels must match beta_ladder replica count {replica_count}"
        ));
    }
    for state in states {
        validate_spin_state(state, n_spins)?;
    }
    validate_labels(labels)
}

fn validate_label_pair(
    label_pair: &[usize],
    replica_count: usize,
) -> Result<(usize, usize), String> {
    if label_pair.len() != 2 {
        return Err("label_pair must contain exactly two adjacent labels".to_string());
    }
    let left = label_pair[0];
    let right = label_pair[1];
    if right != left + 1 || right >= replica_count {
        return Err("label_pair must contain adjacent beta-label indices".to_string());
    }
    Ok((left, right))
}

fn label_position(labels: &[usize], label: usize) -> Result<usize, String> {
    labels
        .iter()
        .position(|value| *value == label)
        .ok_or_else(|| format!("label {label} missing from labels"))
}

fn swap_labels(labels: &[usize], pair: (usize, usize)) -> Result<Vec<usize>, String> {
    let mut updated = labels.to_vec();
    let left_pos = label_position(labels, pair.0)?;
    let right_pos = label_position(labels, pair.1)?;
    updated.swap(left_pos, right_pos);
    Ok(updated)
}

fn normal_cdf(value: f64) -> f64 {
    (0.5 * libm::erfc(-value / SQRT_2)).clamp(1e-300, 1.0)
}

fn acceptance_probability(log_ratio: f64) -> f64 {
    if log_ratio >= 0.0 {
        1.0
    } else if log_ratio < -745.0 {
        0.0
    } else {
        log_ratio.exp()
    }
}

fn next_uniform(rng_state: &mut u64) -> f64 {
    *rng_state = rng_state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let bits = *rng_state >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn draw_uniforms(rng_state: &mut u64, count: usize) -> Vec<f64> {
    (0..count).map(|_| next_uniform(rng_state)).collect()
}

fn draw_uniforms_into(rng_state: &mut u64, output: &mut [f64]) {
    for value in output {
        *value = next_uniform(rng_state);
    }
}
