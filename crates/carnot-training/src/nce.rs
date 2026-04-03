//! Noise Contrastive Estimation (NCE) training algorithm.
//!
//! **Researcher summary:**
//! NCE trains an energy-based model by framing density estimation as binary
//! classification between real data and noise samples drawn from a known
//! distribution. The key advantage over CD-k and maximum likelihood is that
//! NCE completely avoids computing the intractable partition function Z.
//!
//! **Detailed explanation for engineers:**
//! Energy-Based Models assign an energy E(x) to every possible input x.
//! The probability of x is:
//!
//! ```text
//! p(x) = exp(-E(x)) / Z
//! ```
//!
//! where Z = integral of exp(-E(x)) over all x (the "partition function").
//! This integral is intractable for continuous, high-dimensional inputs.
//!
//! **How NCE works:**
//!
//! Instead of computing Z directly, NCE sets up a binary classification task:
//!
//! 1. Take a batch of real data samples x_data.
//! 2. Draw an equal number of noise samples x_noise from a known distribution
//!    (e.g., Gaussian). We know the noise density p_noise(x) exactly.
//! 3. Train the energy function so that:
//!    - Real data gets LOW energy, so `exp(-E(x))` is high (classified as "real")
//!    - Noise samples get HIGH energy, so `exp(-E(x))` is low (classified as "noise")
//!
//! The NCE loss function is:
//!
//! ```text
//! L = -mean(log sigmoid(-E(x_data))) - mean(log sigmoid(E(x_noise)))
//! ```
//!
//! Where `sigmoid(z) = 1 / (1 + exp(-z))`.
//!
//! **Intuition:** `sigmoid(-E(x))` is the model's probability that x is real data.
//! - For real data, we want `sigmoid(-E(x))` to be high, so E(x) should be low.
//! - For noise, we want `sigmoid(E(x))` to be high, so E(x) should be high.
//!
//! **Why this avoids the partition function:**
//! The classifier only needs to compare E(x) to a threshold - it never needs
//! to normalize over all possible x. The noise distribution provides the
//! "reference" that Z would otherwise provide.
//!
//! **Relationship to other training methods:**
//! - CD-k: Requires MCMC sampling during training (expensive, biased).
//! - Score matching: Matches gradients rather than densities.
//! - NCE: Reduces to classification - conceptually simplest, and the loss
//!   gradients are cheap to compute.
//!
//! **Reference:** Gutmann & Hyvarinen (2010), "Noise-Contrastive Estimation
//! of Unnormalized Statistical Models, with Applications to Natural Image
//! Statistics."
//!
//! Spec: REQ-TRAIN-003

use carnot_core::{EnergyFunction, Float};
use ndarray::{Array1, ArrayView2};

/// Configuration for Noise Contrastive Estimation.
///
/// **Detailed explanation for engineers:**
/// `noise_scale` controls the standard deviation of the Gaussian noise
/// distribution used to generate negative (noise) samples. A good noise
/// distribution should overlap somewhat with the data distribution -
/// if noise is too far from the data, the classifier task becomes trivial
/// and the model learns nothing useful.
///
/// `num_noise_samples_per_data` controls how many noise samples to draw
/// per real data sample. Gutmann & Hyvarinen (2010) show that more noise
/// samples reduce the variance of the NCE gradient estimate. A typical
/// value is 1 (equal number of noise and data samples).
#[derive(Debug, Clone)]
pub struct NceConfig {
    /// Standard deviation of the Gaussian noise distribution.
    /// Controls how "spread out" the noise samples are.
    pub noise_scale: Float,
    /// Number of noise samples per data sample (ratio).
    /// Higher values reduce gradient variance but increase compute cost.
    pub num_noise_samples_per_data: usize,
}

impl Default for NceConfig {
    fn default() -> Self {
        Self {
            noise_scale: 1.0,
            num_noise_samples_per_data: 1,
        }
    }
}

/// Compute NCE loss given an energy function, a data batch, and a noise batch.
///
/// **Researcher summary:**
/// L = -mean(log sigmoid(-E(x_data))) - mean(log sigmoid(E(x_noise)))
///
/// **Detailed explanation for engineers:**
/// This function computes the NCE loss for one training step. The caller
/// is responsible for generating the noise batch (this makes the function
/// deterministic and testable).
///
/// **Step by step:**
///
/// 1. Compute E(x) for every real data sample to get data_energies
/// 2. Compute E(x) for every noise sample to get noise_energies
/// 3. For each data sample: loss_data = -log(sigmoid(-E(x)))
///    This pushes E(x) down for real data.
/// 4. For each noise sample: loss_noise = -log(sigmoid(E(x)))
///    This pushes E(x) up for noise.
/// 5. Total loss = mean(loss_data) + mean(loss_noise)
///
/// **Numerical stability:**
/// We use the identity `log(sigmoid(z)) = -softplus(-z) = -log(1 + exp(-z))`
/// to avoid numerical overflow when z is large. Specifically:
///
/// - `log(sigmoid(-E)) = -softplus(E)  = -log(1 + exp(E))`
/// - `log(sigmoid(E))  = -softplus(-E) = -log(1 + exp(-E))`
///
/// The softplus function is numerically stable for all inputs.
///
/// # Arguments
///
/// * `energy_fn` - The energy function to train. Must implement `EnergyFunction`.
/// * `data_batch` - Real data samples, shape (batch_size, dim).
/// * `noise_batch` - Noise samples from a known distribution, shape (num_noise, dim).
///
/// # Returns
///
/// Scalar NCE loss value.
///
/// Spec: REQ-TRAIN-003
pub fn nce_loss(
    energy_fn: &dyn EnergyFunction,
    data_batch: &ArrayView2<Float>,
    noise_batch: &ArrayView2<Float>,
) -> Float {
    let data_size = data_batch.nrows();
    let noise_size = noise_batch.nrows();

    // Step 1: Compute energies for all data samples.
    // energy_batch returns an Array1<Float> with one energy per sample.
    let data_energies: Array1<Float> = energy_fn.energy_batch(data_batch);

    // Step 2: Compute energies for all noise samples.
    let noise_energies: Array1<Float> = energy_fn.energy_batch(noise_batch);

    // Step 3: Compute data term: -mean(log sigmoid(-E(x_data)))
    //
    // log sigmoid(-E) = -softplus(E) = -log(1 + exp(E))
    //
    // We negate this for the loss: loss_data_i = softplus(E(x_data_i))
    // Then mean over the batch.
    let data_loss: Float =
        data_energies.iter().map(|&e| softplus(e)).sum::<Float>() / (data_size as Float);

    // Step 4: Compute noise term: -mean(log sigmoid(E(x_noise)))
    //
    // log sigmoid(E) = -softplus(-E) = -log(1 + exp(-E))
    //
    // We negate this for the loss: loss_noise_i = softplus(-E(x_noise_i))
    // Then mean over the noise samples.
    let noise_loss: Float =
        noise_energies.iter().map(|&e| softplus(-e)).sum::<Float>() / (noise_size as Float);

    // Step 5: Total NCE loss
    data_loss + noise_loss
}

/// Numerically stable softplus: softplus(x) = log(1 + exp(x)).
///
/// **Detailed explanation for engineers:**
/// For large positive x, `exp(x)` overflows. But `log(1 + exp(x))` is
/// approximately x when x is large (because `exp(x) >> 1`). So we use:
///
/// - If x > 20: `softplus(x)` is approximately x
/// - Otherwise: `softplus(x) = log(1 + exp(x))`
///
/// The threshold of 20 is safe because `exp(20)` is about 4.85e8 and
/// `log(1 + 4.85e8)` is about 20.0 to high precision.
fn softplus(x: Float) -> Float {
    if x > 20.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::Float;
    use ndarray::{array, Array1, ArrayView1};

    /// Simple quadratic energy: E(x) = 0.5 * ||x||^2.
    ///
    /// This models a standard Gaussian distribution p(x) ~ exp(-0.5 * ||x||^2).
    /// Data near the origin has low energy; data far away has high energy.
    struct QuadraticEnergy;
    impl EnergyFunction for QuadraticEnergy {
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

    /// Energy function that assigns constant energy everywhere.
    /// This is a "useless" model that cannot distinguish data from noise.
    struct ConstantEnergy {
        value: Float,
    }
    impl EnergyFunction for ConstantEnergy {
        fn energy(&self, _x: &ArrayView1<Float>) -> Float {
            self.value
        }
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            Array1::zeros(x.len())
        }
        fn input_dim(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_nce_loss_is_finite() {
        // REQ-TRAIN-003: NCE loss should be finite for valid inputs.
        // A basic sanity check that the computation doesn't produce NaN or Inf.
        let model = QuadraticEnergy;
        let data = array![[0.1, 0.2], [0.3, -0.1], [-0.2, 0.15]];
        let noise = array![[3.0, 2.0], [-2.5, 1.5], [1.0, -3.0]];

        let loss = nce_loss(&model, &data.view(), &noise.view());
        assert!(loss.is_finite(), "NCE loss should be finite, got {loss}");
    }

    #[test]
    fn test_nce_loss_non_negative() {
        // REQ-TRAIN-003: NCE loss is a sum of softplus terms, which are always >= 0.
        let model = QuadraticEnergy;
        let data = array![[0.5, 0.5], [-0.5, -0.5]];
        let noise = array![[5.0, 5.0], [-5.0, -5.0]];

        let loss = nce_loss(&model, &data.view(), &noise.view());
        assert!(loss >= 0.0, "NCE loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_nce_good_model_lower_loss_than_bad() {
        // REQ-TRAIN-003: A model that correctly assigns low energy to data and
        // high energy to noise should have lower NCE loss than a model that
        // does the opposite.
        //
        // QuadraticEnergy: E(x) = 0.5 * ||x||^2
        // Data near origin → low energy (good)
        // Noise far from origin → high energy (good)
        //
        // ConstantEnergy(0): E(x) = 0 everywhere → cannot discriminate at all.
        // NCE loss for constant energy = softplus(0) + softplus(0) = 2*ln(2) ≈ 1.386
        let data_near_origin = array![[0.01, 0.02], [0.0, 0.0], [-0.01, 0.01]];
        let noise_far_away = array![[10.0, 10.0], [-10.0, 8.0], [7.0, -9.0]];

        let good_model = QuadraticEnergy;
        let bad_model = ConstantEnergy { value: 0.0 };

        let good_loss = nce_loss(
            &good_model,
            &data_near_origin.view(),
            &noise_far_away.view(),
        );
        let bad_loss = nce_loss(&bad_model, &data_near_origin.view(), &noise_far_away.view());

        assert!(
            good_loss < bad_loss,
            "Good model should have lower NCE loss: good={good_loss} < bad={bad_loss}"
        );
    }

    #[test]
    fn test_nce_noise_gets_high_energy() {
        // REQ-TRAIN-003: For a well-matched model and data, noise samples that
        // are far from the data distribution should receive high energy.
        let model = QuadraticEnergy;
        let noise_far = array![[100.0, 100.0]];
        let noise_energies = model.energy_batch(&noise_far.view());

        // E(x) = 0.5 * (100^2 + 100^2) = 10000
        assert!(
            noise_energies[0] > 1000.0,
            "Far-away noise should have high energy, got {}",
            noise_energies[0]
        );
    }

    #[test]
    fn test_nce_perfect_separation_low_loss() {
        // REQ-TRAIN-003: When data has very low energy and noise has very high
        // energy, the NCE loss should approach 0.
        //
        // For data at origin: E = 0, loss_data = softplus(0) = ln(2) ≈ 0.693
        // For noise at (100,100): E = 10000, loss_noise = softplus(-10000) ≈ 0
        // Total ≈ 0.693 (bounded below by ln(2) for any model)
        //
        // But if we use a model that gives E(data) = -1000 and E(noise) = 1000,
        // then loss_data = softplus(-1000) ≈ 0, loss_noise = softplus(-1000) ≈ 0.
        // Total ≈ 0. We can approach this with a scaled model.
        let _model = ConstantEnergy { value: -100.0 };
        let _data = array![[0.0, 0.0]];
        // For constant energy -100:
        // loss_data = softplus(-100) ≈ 0
        // loss_noise = softplus(100) = 100 (large!)
        // So constant energy doesn't help. Let's just verify the math works.

        let good_model = QuadraticEnergy;
        let data_origin = array![[0.0, 0.0], [0.001, 0.0]];
        let noise_far = array![[50.0, 50.0], [-50.0, 50.0]];

        let loss = nce_loss(&good_model, &data_origin.view(), &noise_far.view());
        // data energy ≈ 0 → softplus(0) = ln(2) ≈ 0.693
        // noise energy ≈ 2500 → softplus(-2500) ≈ 0
        // loss ≈ 0.693
        assert!(loss.is_finite());
        // The noise term should be negligible, so loss ≈ ln(2)
        assert!(
            (loss - (2.0_f64.ln() as Float)).abs() < 0.01,
            "Loss for well-separated data/noise should be close to ln(2), got {loss}"
        );
    }

    #[test]
    fn test_nce_softplus_stability() {
        // REQ-TRAIN-003: softplus should not overflow for large inputs.
        assert!((softplus(0.0) - (2.0 as Float).ln()).abs() < 1e-6);
        assert!((softplus(100.0) - 100.0).abs() < 1e-6);
        assert!(softplus(-100.0) < 1e-40);
        assert!(softplus(1.0).is_finite());
    }

    #[test]
    fn test_nce_config_default() {
        // REQ-TRAIN-003: Default config should have sensible values.
        let config = NceConfig::default();
        assert!(config.noise_scale > 0.0);
        assert!(config.num_noise_samples_per_data >= 1);
    }
}
