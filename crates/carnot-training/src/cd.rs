//! Contrastive Divergence (CD-k) training algorithm.
//!
//! Spec: REQ-TRAIN-001

use carnot_core::{EnergyFunction, Float};
use carnot_samplers::{LangevinSampler, Sampler};
use ndarray::{Array1, Array2, ArrayView2};

/// Configuration for CD-k training.
#[derive(Debug, Clone)]
pub struct CdConfig {
    /// Number of MCMC steps for negative phase.
    pub k: usize,
    /// Sampler step size for negative phase.
    pub sampler_step_size: Float,
}

impl Default for CdConfig {
    fn default() -> Self {
        Self {
            k: 1,
            sampler_step_size: 0.01,
        }
    }
}

/// Compute CD-k loss: mean(E(x_pos)) - mean(E(x_neg))
///
/// Returns (loss, positive energies, negative samples).
///
/// Spec: REQ-TRAIN-001
pub fn cd_loss(
    energy_fn: &dyn EnergyFunction,
    batch: &ArrayView2<Float>,
    config: &CdConfig,
) -> (Float, Array1<Float>, Array2<Float>) {
    let sampler = LangevinSampler::new(config.sampler_step_size);
    let batch_size = batch.nrows();

    // Positive phase: energy on data
    let pos_energies = energy_fn.energy_batch(batch);

    // Negative phase: run k steps of MCMC from data
    let mut neg_samples = batch.to_owned();
    for row_idx in 0..batch_size {
        let init = neg_samples.row(row_idx).to_owned();
        let sample = sampler.sample(energy_fn, &init, config.k);
        neg_samples.row_mut(row_idx).assign(&sample);
    }
    let neg_energies = energy_fn.energy_batch(&neg_samples.view());

    let loss = pos_energies.mean().unwrap_or(0.0) - neg_energies.mean().unwrap_or(0.0);

    (loss, pos_energies, neg_samples)
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::Float;
    use ndarray::{array, Array1, ArrayView1};

    struct SimpleEnergy;
    impl EnergyFunction for SimpleEnergy {
        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            0.5 * x.dot(x)
        }
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            x.to_owned()
        }
        fn input_dim(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_cd_loss_computes() {
        // SCENARIO-TRAIN-001: CD-1 training step
        let model = SimpleEnergy;
        let batch = array![[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]];
        let config = CdConfig::default();

        let (loss, pos_energies, neg_samples) = cd_loss(&model, &batch.view(), &config);

        assert!(loss.is_finite(), "CD loss should be finite, got {loss}");
        assert_eq!(pos_energies.len(), 3);
        assert_eq!(neg_samples.shape(), &[3, 2]);
    }

    #[test]
    fn test_cd_negative_samples_differ() {
        // SCENARIO-TRAIN-001: negative samples should differ from positive
        let model = SimpleEnergy;
        let batch = array![[5.0, 5.0], [5.0, 5.0]];
        let config = CdConfig {
            k: 10,
            sampler_step_size: 0.1,
        };

        let (_, _, neg_samples) = cd_loss(&model, &batch.view(), &config);

        // Negative samples should have moved from the initial position
        let diff = &neg_samples - &batch;
        let total_diff: Float = diff.mapv(|v| v.abs()).sum();
        assert!(
            total_diff > 0.01,
            "Negative samples should differ from positive after k steps"
        );
    }
}
