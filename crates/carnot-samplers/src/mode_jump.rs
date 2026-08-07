//! Fixed categorical mode-jump Metropolis-Hastings sampler.
//!
//! The kernel is deliberately finite and explicit. Exp6166/Exp6180 froze a
//! categorical target and local-plus-cross-mode proposal table; this module
//! ports that transition only, with deterministic state replay for PyO3 parity.
//!
//! Spec: REQ-SAMPLE-6194, SCENARIO-SAMPLE-6194-EXACT-TRANSITION-PARITY

use std::collections::HashSet;

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;
const STATE_SCHEMA: &str = "mode_jump_state_v1";
const NORMALIZATION_TOLERANCE: f64 = 1e-12;

/// Frozen finite categorical target and proposal table.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeJumpConfig {
    pub labels: Vec<String>,
    pub target_probabilities: Vec<f64>,
    pub proposal_probabilities: Vec<Vec<f64>>,
}

impl ModeJumpConfig {
    /// Validate a finite target and proposal table before sampling.
    pub fn new(
        labels: Vec<String>,
        target_probabilities: Vec<f64>,
        proposal_probabilities: Vec<Vec<f64>>,
    ) -> Result<Self, String> {
        if labels.is_empty() {
            return Err("labels must be non-empty".to_string());
        }
        let n = labels.len();
        if target_probabilities.len() != n {
            return Err(format!(
                "target_probabilities length must match labels: expected {n}, got {}",
                target_probabilities.len()
            ));
        }
        if proposal_probabilities.len() != n {
            return Err(format!(
                "proposal row count must match labels: expected {n}, got {}",
                proposal_probabilities.len()
            ));
        }
        let mut seen = HashSet::with_capacity(n);
        for label in &labels {
            if label.is_empty() || label.contains('|') {
                return Err("labels must be non-empty and must not contain '|'".to_string());
            }
            if !seen.insert(label.clone()) {
                return Err("labels must be unique".to_string());
            }
        }
        validate_probability_vector(
            &target_probabilities,
            "target_probabilities",
            true,
            NORMALIZATION_TOLERANCE,
        )?;
        for (row_index, row) in proposal_probabilities.iter().enumerate() {
            if row.len() != n {
                return Err(format!(
                    "proposal row {row_index} length must match labels: expected {n}, got {}",
                    row.len()
                ));
            }
            validate_probability_vector(
                row,
                &format!("proposal row {row_index}"),
                false,
                NORMALIZATION_TOLERANCE,
            )?;
        }
        for i in 0..n {
            for j in 0..n {
                let forward = proposal_probabilities[i][j] > 0.0;
                let reverse = proposal_probabilities[j][i] > 0.0;
                if forward != reverse {
                    return Err("proposal support must be symmetric for MH correction".to_string());
                }
            }
        }
        Ok(Self {
            labels,
            target_probabilities,
            proposal_probabilities,
        })
    }
}

/// Serializable seeded sampler state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModeJumpState {
    pub current_label: String,
    pub rng_state: u64,
    pub step: usize,
    pub accepted_count: usize,
}

impl ModeJumpState {
    /// Build a state after validating local counters and label syntax.
    pub fn new(
        current_label: String,
        rng_state: u64,
        step: usize,
        accepted_count: usize,
    ) -> Result<Self, String> {
        if current_label.is_empty() || current_label.contains('|') {
            return Err("current_label must be non-empty and must not contain '|'".to_string());
        }
        if accepted_count > step {
            return Err("accepted_count must be less than or equal to step".to_string());
        }
        Ok(Self {
            current_label,
            rng_state,
            step,
            accepted_count,
        })
    }

    /// Serialize to a compact ASCII schema that can be checked before restore.
    pub fn serialize(&self) -> String {
        format!(
            "{STATE_SCHEMA}|{}|{}|{}|{}",
            self.current_label, self.rng_state, self.step, self.accepted_count
        )
    }

    /// Deserialize the compact state string. Label membership is checked by the core.
    pub fn deserialize(serialized: &str) -> Result<Self, String> {
        let fields: Vec<&str> = serialized.split('|').collect();
        if fields.len() != 5 || fields[0] != STATE_SCHEMA {
            return Err("serialized state must use mode_jump_state_v1 schema".to_string());
        }
        let rng_state = fields[2]
            .parse::<u64>()
            .map_err(|_| "serialized rng_state must be u64".to_string())?;
        let step = fields[3]
            .parse::<usize>()
            .map_err(|_| "serialized step must be usize".to_string())?;
        let accepted_count = fields[4]
            .parse::<usize>()
            .map_err(|_| "serialized accepted_count must be usize".to_string())?;
        Self::new(fields[1].to_string(), rng_state, step, accepted_count)
    }
}

/// One Metropolis-Hastings transition with all deterministic diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeJumpStepOutcome {
    pub state_before: ModeJumpState,
    pub state: ModeJumpState,
    pub proposal_uniform: f64,
    pub proposed_label: String,
    pub acceptance_uniform: f64,
    pub current_energy: f64,
    pub proposed_energy: f64,
    pub proposal_log_forward: f64,
    pub proposal_log_reverse: f64,
    pub log_acceptance: f64,
    pub acceptance_probability: f64,
    pub accepted: bool,
}

/// Empirical frequency for one label.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeJumpFrequency {
    pub label: String,
    pub count: usize,
    pub frequency: f64,
    pub target_probability: f64,
}

/// Long-run diagnostics used by Exp6194 parity checks.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeJumpRunSummary {
    pub sample_count: usize,
    pub burn_in: usize,
    pub frequencies: Vec<ModeJumpFrequency>,
    pub total_variation_to_target: f64,
    pub kl_target_to_empirical: f64,
    pub accepted_count: usize,
    pub attempted_count: usize,
    pub acceptance_rate: f64,
    pub lag1_autocorrelation: f64,
    pub integrated_autocorrelation_time: f64,
    pub effective_sample_size: f64,
    pub final_state: ModeJumpState,
}

impl ModeJumpRunSummary {
    /// Return one empirical frequency by label.
    pub fn frequency(&self, label: &str) -> Option<f64> {
        self.frequencies
            .iter()
            .find(|row| row.label == label)
            .map(|row| row.frequency)
    }
}

/// Exact finite-state mode-jump transition kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeJumpCore {
    pub config: ModeJumpConfig,
}

impl ModeJumpCore {
    pub fn new(config: ModeJumpConfig) -> Self {
        Self { config }
    }

    /// Compute `E(label) = -log pi(label)`.
    pub fn energy(&self, label: &str) -> Result<f64, String> {
        let index = self.label_index(label)?;
        Ok(-self.config.target_probabilities[index].ln())
    }

    /// Return `q(proposed | current)` from the frozen proposal table.
    pub fn proposal_probability(&self, current: &str, proposed: &str) -> Result<f64, String> {
        let current_index = self.label_index(current)?;
        let proposed_index = self.label_index(proposed)?;
        Ok(self.config.proposal_probabilities[current_index][proposed_index])
    }

    /// Restore a serialized state and validate it against this configuration.
    pub fn state_from_serialized(&self, serialized: &str) -> Result<ModeJumpState, String> {
        let state = ModeJumpState::deserialize(serialized)?;
        self.validate_state(&state)?;
        Ok(state)
    }

    /// Run one deterministic MH transition from the state's RNG.
    pub fn step_trace(&self, state: &ModeJumpState) -> Result<ModeJumpStepOutcome, String> {
        self.validate_state(state)?;
        let current_index = self.label_index(&state.current_label)?;
        let mut rng_state = state.rng_state;
        let proposal_uniform = next_uniform(&mut rng_state);
        let proposed_index = draw_index(
            &self.config.proposal_probabilities[current_index],
            proposal_uniform,
        );
        let acceptance_uniform = next_uniform(&mut rng_state);
        let q_forward = self.config.proposal_probabilities[current_index][proposed_index];
        let q_reverse = self.config.proposal_probabilities[proposed_index][current_index];
        if q_forward <= 0.0 || q_reverse <= 0.0 {
            return Err(
                "proposal support must include forward and reverse probabilities".to_string(),
            );
        }
        let current_probability = self.config.target_probabilities[current_index];
        let proposed_probability = self.config.target_probabilities[proposed_index];
        let current_energy = -current_probability.ln();
        let proposed_energy = -proposed_probability.ln();
        let proposal_log_forward = q_forward.ln();
        let proposal_log_reverse = q_reverse.ln();
        let log_acceptance = proposed_probability.ln() - current_probability.ln()
            + proposal_log_reverse
            - proposal_log_forward;
        let acceptance_probability = acceptance_probability(log_acceptance);
        let accepted = log_acceptance >= 0.0 || acceptance_uniform.ln() < log_acceptance;
        let proposed_label = self.config.labels[proposed_index].clone();
        let state_after = ModeJumpState {
            current_label: if accepted {
                proposed_label.clone()
            } else {
                state.current_label.clone()
            },
            rng_state,
            step: state
                .step
                .checked_add(1)
                .ok_or_else(|| "step counter overflow".to_string())?,
            accepted_count: state
                .accepted_count
                .checked_add(usize::from(accepted))
                .ok_or_else(|| "accepted_count overflow".to_string())?,
        };
        Ok(ModeJumpStepOutcome {
            state_before: state.clone(),
            state: state_after,
            proposal_uniform,
            proposed_label,
            acceptance_uniform,
            current_energy,
            proposed_energy,
            proposal_log_forward,
            proposal_log_reverse,
            log_acceptance,
            acceptance_probability,
            accepted,
        })
    }

    /// Run a bounded chain and retain samples after burn-in.
    pub fn run(
        &self,
        state: &ModeJumpState,
        n_steps: usize,
        burn_in: usize,
    ) -> Result<ModeJumpRunSummary, String> {
        if n_steps == 0 {
            return Err("n_steps must be positive".to_string());
        }
        if burn_in >= n_steps {
            return Err("burn_in must be smaller than n_steps".to_string());
        }
        self.validate_state(state)?;
        let sample_count = n_steps - burn_in;
        let mut current = state.clone();
        let mut counts = vec![0_usize; self.config.labels.len()];
        let mut accepted = 0_usize;
        let mut indicator = Vec::with_capacity(sample_count);
        for step_index in 0..n_steps {
            let outcome = self.step_trace(&current)?;
            if outcome.accepted {
                accepted += 1;
            }
            current = outcome.state;
            if step_index >= burn_in {
                let index = self.label_index(&current.current_label)?;
                counts[index] += 1;
                indicator.push(if index == 0 { 1.0 } else { 0.0 });
            }
        }
        let frequencies: Vec<ModeJumpFrequency> = self
            .config
            .labels
            .iter()
            .enumerate()
            .map(|(index, label)| ModeJumpFrequency {
                label: label.clone(),
                count: counts[index],
                frequency: counts[index] as f64 / sample_count as f64,
                target_probability: self.config.target_probabilities[index],
            })
            .collect();
        let total_variation_to_target = frequencies
            .iter()
            .map(|row| (row.frequency - row.target_probability).abs())
            .sum::<f64>()
            * 0.5;
        let kl_target_to_empirical = frequencies
            .iter()
            .map(|row| {
                if row.target_probability == 0.0 {
                    0.0
                } else if row.frequency == 0.0 {
                    f64::INFINITY
                } else {
                    row.target_probability * (row.target_probability / row.frequency).ln()
                }
            })
            .sum();
        let (lag1_autocorrelation, integrated_autocorrelation_time, effective_sample_size) =
            quality_from_indicator(&indicator);
        Ok(ModeJumpRunSummary {
            sample_count,
            burn_in,
            frequencies,
            total_variation_to_target,
            kl_target_to_empirical,
            accepted_count: accepted,
            attempted_count: n_steps,
            acceptance_rate: accepted as f64 / n_steps as f64,
            lag1_autocorrelation,
            integrated_autocorrelation_time,
            effective_sample_size,
            final_state: current,
        })
    }

    fn validate_state(&self, state: &ModeJumpState) -> Result<(), String> {
        if state.accepted_count > state.step {
            return Err("accepted_count must be less than or equal to step".to_string());
        }
        self.label_index(&state.current_label)?;
        Ok(())
    }

    fn label_index(&self, label: &str) -> Result<usize, String> {
        self.config
            .labels
            .iter()
            .position(|candidate| candidate == label)
            .ok_or_else(|| format!("label not in mode-jump configuration: {label}"))
    }
}

fn validate_probability_vector(
    values: &[f64],
    name: &str,
    require_strictly_positive: bool,
    tolerance: f64,
) -> Result<(), String> {
    if values.is_empty() {
        return Err(format!("{name} must be non-empty"));
    }
    let mut sum = 0.0;
    for value in values {
        if !value.is_finite() {
            return Err(format!("{name} values must be finite"));
        }
        if *value < 0.0 || (require_strictly_positive && *value <= 0.0) {
            return Err(format!("{name} values must be positive on support"));
        }
        sum += *value;
    }
    if (sum - 1.0).abs() > tolerance {
        return Err(format!("{name} must sum to 1.0"));
    }
    Ok(())
}

fn next_uniform(rng_state: &mut u64) -> f64 {
    *rng_state = rng_state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let bits = *rng_state >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
}

fn draw_index(probabilities: &[f64], uniform: f64) -> usize {
    let mut cumulative = 0.0;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative += *probability;
        if uniform < cumulative {
            return index;
        }
    }
    probabilities.len() - 1
}

fn acceptance_probability(log_acceptance: f64) -> f64 {
    if log_acceptance >= 0.0 {
        1.0
    } else if log_acceptance < -745.0 {
        0.0
    } else {
        log_acceptance.exp()
    }
}

fn quality_from_indicator(values: &[f64]) -> (f64, f64, f64) {
    if values.len() < 2 {
        return (0.0, 1.0, values.len() as f64);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let denom = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>();
    if denom == 0.0 {
        return (0.0, 1.0, values.len() as f64);
    }
    let lag1 = autocorrelation(values, mean, denom, 1);
    let mut positive_sum = 0.0;
    let max_lag = values.len().saturating_sub(1).min(200);
    for lag in 1..=max_lag {
        let rho = autocorrelation(values, mean, denom, lag);
        if rho <= 0.0 {
            break;
        }
        positive_sum += rho;
    }
    let iact = (1.0 + 2.0 * positive_sum).max(1.0);
    let ess = values.len() as f64 / iact;
    (lag1, iact, ess)
}

fn autocorrelation(values: &[f64], mean: f64, denom: f64, lag: usize) -> f64 {
    let mut numer = 0.0;
    for index in lag..values.len() {
        numer += (values[index] - mean) * (values[index - lag] - mean);
    }
    numer / denom
}
