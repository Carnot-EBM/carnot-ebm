//! Exact bounded block heat-bath transitions for finite Ising models.
//!
//! The caller supplies a complete spin partition. Each transition chooses one
//! block and enumerates only that block's assignments. This keeps the target
//! distribution exact while the full state space can remain too large to
//! enumerate.
//!
//! Spec: REQ-SAMPLER-6612, REQ-RUSTPY-6612

const LCG_A: u64 = 6364136223846793005;
const LCG_C: u64 = 1442695040888963407;
const MAX_BLOCK_SIZE: usize = 16;

/// Validated Ising model and complete spin partition.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralKBlockConfig {
    pub couplings: Vec<Vec<f64>>,
    pub fields: Vec<f64>,
    pub temperature: f64,
    pub blocks: Vec<Vec<usize>>,
}

impl SpectralKBlockConfig {
    /// Validate finite model data and exact one-block-per-spin membership.
    pub fn new(
        couplings: Vec<Vec<f64>>,
        fields: Vec<f64>,
        temperature: f64,
        blocks: Vec<Vec<usize>>,
    ) -> Result<Self, String> {
        if couplings.is_empty() || couplings.iter().any(|row| row.len() != couplings.len()) {
            return Err("couplings must be square and non-empty".to_string());
        }
        let n_spins = couplings.len();
        if fields.len() != n_spins {
            return Err(format!(
                "fields length must match couplings dimension: expected {n_spins}, got {}",
                fields.len()
            ));
        }
        if !temperature.is_finite() || temperature <= 0.0 {
            return Err("temperature must be finite and positive".to_string());
        }
        if couplings
            .iter()
            .flatten()
            .chain(fields.iter())
            .any(|value| !value.is_finite())
        {
            return Err("couplings and fields must be finite".to_string());
        }
        for (i, row) in couplings.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                if (*value - couplings[j][i]).abs() > 1.0e-12 {
                    return Err("couplings must be symmetric".to_string());
                }
            }
        }
        if blocks.is_empty()
            || blocks
                .iter()
                .any(|block| block.is_empty() || block.len() > MAX_BLOCK_SIZE)
        {
            return Err(format!(
                "partition blocks must be non-empty and contain at most {MAX_BLOCK_SIZE} spins"
            ));
        }
        let mut counts = vec![0_usize; n_spins];
        for index in blocks.iter().flatten() {
            if *index >= n_spins {
                return Err("partition spin index is out of range".to_string());
            }
            counts[*index] += 1;
        }
        if counts.iter().any(|count| *count != 1) {
            return Err("partition must contain every spin exactly once".to_string());
        }
        Ok(Self {
            couplings,
            fields,
            temperature,
            blocks,
        })
    }

    pub fn n_spins(&self) -> usize {
        self.fields.len()
    }
}

/// Seeded chain state that can cross the Python/Rust boundary without loss.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpectralKBlockState {
    pub spins: Vec<i8>,
    pub rng_state: u64,
    pub transition: usize,
    pub spins_updated: usize,
}

impl SpectralKBlockState {
    pub fn new(
        spins: Vec<i8>,
        rng_state: u64,
        transition: usize,
        spins_updated: usize,
    ) -> Result<Self, String> {
        validate_spins(&spins)?;
        Ok(Self {
            spins,
            rng_state,
            transition,
            spins_updated,
        })
    }
}

/// Flat retained samples plus exact work counters and restart state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpectralKBlockOutcome {
    pub samples: Vec<i8>,
    pub final_state: SpectralKBlockState,
    pub transitions: usize,
    pub spins_updated: usize,
}

/// Deterministic exact block heat-bath core.
#[derive(Clone, Debug, PartialEq)]
pub struct SpectralKBlockCore {
    pub config: SpectralKBlockConfig,
}

impl SpectralKBlockCore {
    pub fn new(config: SpectralKBlockConfig) -> Self {
        Self { config }
    }

    /// Compute the shared energy convention `-0.5*s^T*J*s - h^T*s`.
    pub fn energy(&self, spins: &[i8]) -> Result<f64, String> {
        self.validate_state(spins)?;
        let mut pair = 0.0;
        let mut field = 0.0;
        for i in 0..self.config.n_spins() {
            let spin_i = f64::from(spins[i]);
            field += self.config.fields[i] * spin_i;
            for (j, spin_j) in spins.iter().enumerate() {
                pair += spin_i * self.config.couplings[i][j] * f64::from(*spin_j);
            }
        }
        Ok(-0.5 * pair - field)
    }

    /// Run charged burn-in transitions followed by retained transitions.
    pub fn run_chain(
        &self,
        initial: &SpectralKBlockState,
        burn_in: usize,
        retained_samples: usize,
    ) -> Result<SpectralKBlockOutcome, String> {
        self.validate_state(&initial.spins)?;
        if retained_samples == 0 {
            return Err("retained_samples must be positive".to_string());
        }
        let total = burn_in
            .checked_add(retained_samples)
            .ok_or_else(|| "transition count overflow".to_string())?;
        let output_len = retained_samples
            .checked_mul(self.config.n_spins())
            .ok_or_else(|| "sample allocation overflow".to_string())?;
        let mut state = initial.clone();
        let mut samples = Vec::with_capacity(output_len);
        let start_transition = state.transition;
        let start_spins_updated = state.spins_updated;
        for offset in 0..total {
            self.step_in_place(&mut state)?;
            if offset >= burn_in {
                samples.extend_from_slice(&state.spins);
            }
        }
        Ok(SpectralKBlockOutcome {
            samples,
            transitions: state.transition - start_transition,
            spins_updated: state.spins_updated - start_spins_updated,
            final_state: state,
        })
    }

    fn validate_state(&self, spins: &[i8]) -> Result<(), String> {
        validate_spins(spins)?;
        if spins.len() != self.config.n_spins() {
            return Err(format!(
                "spin state length must be {}",
                self.config.n_spins()
            ));
        }
        Ok(())
    }

    fn step_in_place(&self, state: &mut SpectralKBlockState) -> Result<(), String> {
        let block_uniform = next_uniform(&mut state.rng_state);
        let block_index = ((block_uniform * self.config.blocks.len() as f64) as usize)
            .min(self.config.blocks.len() - 1);
        let draw_uniform = next_uniform(&mut state.rng_state);
        let block = &self.config.blocks[block_index];
        let assignment_count = 1_usize << block.len();
        let mut log_weights = Vec::with_capacity(assignment_count);
        for assignment in 0..assignment_count {
            log_weights.push(self.conditional_log_weight(&state.spins, block, assignment));
        }
        let maximum = log_weights
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = log_weights
            .iter()
            .map(|value| (*value - maximum).exp())
            .collect();
        let threshold = draw_uniform * weights.iter().sum::<f64>();
        let mut cumulative = 0.0;
        let mut selected = assignment_count - 1;
        for (assignment, weight) in weights.iter().enumerate() {
            cumulative += weight;
            if threshold < cumulative {
                selected = assignment;
                break;
            }
        }
        for (local, spin_index) in block.iter().enumerate() {
            state.spins[*spin_index] = if (selected >> local) & 1 == 1 { 1 } else { -1 };
        }
        state.transition = state
            .transition
            .checked_add(1)
            .ok_or_else(|| "transition counter overflow".to_string())?;
        state.spins_updated = state
            .spins_updated
            .checked_add(block.len())
            .ok_or_else(|| "spin update counter overflow".to_string())?;
        Ok(())
    }

    fn conditional_log_weight(&self, state: &[i8], block: &[usize], assignment: usize) -> f64 {
        let mut proposed = Vec::with_capacity(block.len());
        for local in 0..block.len() {
            proposed.push(if (assignment >> local) & 1 == 1 {
                1.0
            } else {
                -1.0
            });
        }
        let mut in_block = vec![false; self.config.n_spins()];
        for index in block {
            in_block[*index] = true;
        }
        let mut exponent = 0.0;
        for (local, spin_index) in block.iter().enumerate() {
            let mut outside_field = self.config.fields[*spin_index];
            for j in 0..self.config.n_spins() {
                if !in_block[j] {
                    outside_field += self.config.couplings[*spin_index][j] * f64::from(state[j]);
                }
            }
            exponent += proposed[local] * outside_field;
            for later in (local + 1)..block.len() {
                exponent += self.config.couplings[*spin_index][block[later]]
                    * proposed[local]
                    * proposed[later];
            }
        }
        exponent / self.config.temperature
    }
}

fn validate_spins(spins: &[i8]) -> Result<(), String> {
    if spins.is_empty() || spins.iter().any(|value| *value != -1 && *value != 1) {
        return Err("spins must be a non-empty vector containing only -1 or +1".to_string());
    }
    Ok(())
}

fn next_uniform(rng_state: &mut u64) -> f64 {
    *rng_state = rng_state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
    let bits = *rng_state >> 11;
    (bits as f64) * (1.0 / ((1_u64 << 53) as f64))
}
