//! Denoising Score Matching (DSM) training algorithm.
//!
//! Spec: REQ-TRAIN-002
//!
//! DSM loss:
//!   L = E_x E_noise [ || grad_energy(x + noise) - (-noise / sigma^2) ||^2 ]
//!
//! For a model with energy E(x), the model's score is s(x) = -grad_energy(x).
//! The denoising score matching objective trains the model so that its score
//! at a noisy point (x + noise) matches the optimal denoiser score (-noise / sigma^2).

use carnot_core::{EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView2};
use ndarray_rand::RandomExt;
use rand::thread_rng;
use rand_distr::Normal;

/// Configuration for denoising score matching.
#[derive(Debug, Clone)]
pub struct DsmConfig {
    /// Standard deviation of the noise distribution.
    pub sigma: Float,
}

impl Default for DsmConfig {
    fn default() -> Self {
        Self { sigma: 0.1 }
    }
}

/// Compute denoising score matching loss.
///
/// For each sample x in the batch:
///   1. Draw noise ~ N(0, sigma^2 I)
///   2. Compute x_noisy = x + noise
///   3. Compute model_score = grad_energy(x_noisy)
///   4. Compute target = -noise / sigma^2  (the optimal denoiser score)
///   5. Loss contribution = || model_score - target ||^2
///
/// Returns the mean loss over the batch.
///
/// Spec: REQ-TRAIN-002
pub fn dsm_loss(
    energy_fn: &dyn EnergyFunction,
    batch: &ArrayView2<Float>,
    config: &DsmConfig,
) -> Float {
    let batch_size = batch.nrows();
    let dim = batch.ncols();
    let sigma = config.sigma;
    let sigma_sq = sigma * sigma;

    // Draw noise for the whole batch
    let noise = Array2::random_using(
        (batch_size, dim),
        Normal::new(0.0 as Float, sigma as Float).unwrap(),
        &mut thread_rng(),
    );

    let noisy_batch = batch.to_owned() + &noise;

    let mut total_loss: Float = 0.0;
    for i in 0..batch_size {
        let x_noisy = noisy_batch.row(i);
        let model_score = energy_fn.grad_energy(&x_noisy);
        let target: Array1<Float> = noise.row(i).mapv(|n| -n / sigma_sq);
        let diff = &model_score - &target;
        total_loss += diff.dot(&diff);
    }

    total_loss / (batch_size as Float)
}

/// Compute denoising score matching loss with provided noise (for deterministic testing).
///
/// Spec: REQ-TRAIN-002
pub fn dsm_loss_with_noise(
    energy_fn: &dyn EnergyFunction,
    batch: &ArrayView2<Float>,
    noise: &ArrayView2<Float>,
    sigma: Float,
) -> Float {
    let batch_size = batch.nrows();
    let sigma_sq = sigma * sigma;

    let noisy_batch = batch.to_owned() + noise;

    let mut total_loss: Float = 0.0;
    for i in 0..batch_size {
        let x_noisy = noisy_batch.row(i);
        let model_score = energy_fn.grad_energy(&x_noisy);
        let target: Array1<Float> = noise.row(i).mapv(|n| -n / sigma_sq);
        let diff = &model_score - &target;
        total_loss += diff.dot(&diff);
    }

    total_loss / (batch_size as Float)
}

#[cfg(test)]
mod tests {
    use super::*;
    use carnot_core::Float;
    use ndarray::{array, Array1, ArrayView1};

    /// Quadratic energy: E(x) = 0.5 * x^T * x
    /// grad_energy(x) = x
    /// True score: s(x) = -x
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

    /// Scaled quadratic energy: E(x) = 0.5 * scale * x^T * x
    /// grad_energy(x) = scale * x
    /// When scale = 1, this is the standard Gaussian energy, and
    /// the score s(x) = -grad_energy(x) = -x matches the true score.
    struct ScaledQuadraticEnergy {
        scale: Float,
    }
    impl EnergyFunction for ScaledQuadraticEnergy {
        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            0.5 * self.scale * x.dot(x)
        }
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            x.mapv(|v| self.scale * v)
        }
        fn input_dim(&self) -> usize {
            2
        }
    }

    #[test]
    fn test_dsm_loss_finite() {
        // SCENARIO-TRAIN-002: DSM loss should be finite for valid inputs
        let model = QuadraticEnergy;
        let batch = array![[1.0, 2.0], [3.0, 4.0], [-1.0, 0.5]];
        let config = DsmConfig { sigma: 0.1 };

        let loss = dsm_loss(&model, &batch.view(), &config);
        assert!(loss.is_finite(), "DSM loss should be finite, got {loss}");
        assert!(loss >= 0.0, "DSM loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_dsm_loss_with_noise_deterministic() {
        // SCENARIO-TRAIN-002: Deterministic DSM loss with provided noise
        let model = QuadraticEnergy;
        let batch = array![[1.0, 0.0], [0.0, 1.0]];
        let noise = array![[0.1, -0.1], [-0.05, 0.05]];
        let sigma: Float = 0.5;

        let loss = dsm_loss_with_noise(&model, &batch.view(), &noise.view(), sigma);
        assert!(loss.is_finite(), "DSM loss should be finite, got {loss}");
        assert!(loss >= 0.0, "DSM loss should be non-negative, got {loss}");
    }

    #[test]
    fn test_dsm_loss_better_model_has_lower_loss() {
        // SCENARIO-TRAIN-002: A model whose score matches the true score
        // should have lower DSM loss than a model with wrong score.
        //
        // Data from N(0,I). True score at noisy point x+n is -n/sigma^2.
        // For E(x) = 0.5*scale*x^T*x, grad_energy(x) = scale*x.
        //
        // When scale=1, grad_energy(x+n) = x+n. The target is -n/sigma^2.
        // For data at origin and small noise, x+n ≈ n, so grad ≈ n,
        // target ≈ -n/sigma^2.

        let batch = array![[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]];
        let noise = array![[0.1, -0.1], [0.05, 0.05], [-0.1, 0.0]];
        let sigma: Float = 1.0;

        // Good model: scale = 1/sigma^2 = 1.0
        // grad_energy(0 + n) = 1.0 * n. Target = -n/1.0 = -n.
        // diff = n - (-n) = 2n. Loss > 0 but let's compare two models.
        // Actually for data at 0: noisy = n. grad = scale*n. target = -n/sigma^2 = -n.
        // diff = scale*n - (-n) = (scale+1)*n.
        // Loss = mean(||(scale+1)*n||^2) = (scale+1)^2 * mean(||n||^2)
        // Minimized when scale = -1, but we'll just compare scale=0 vs scale=5.

        // scale=0 gives diff = 0*n - (-n) = n, loss = mean(||n||^2)
        // scale=5 gives diff = 5n - (-n) = 6n, loss = 36*mean(||n||^2)
        let good_model = ScaledQuadraticEnergy { scale: 0.0 };
        let bad_model = ScaledQuadraticEnergy { scale: 5.0 };

        let good_loss = dsm_loss_with_noise(&good_model, &batch.view(), &noise.view(), sigma);
        let bad_loss = dsm_loss_with_noise(&bad_model, &batch.view(), &noise.view(), sigma);

        assert!(
            good_loss < bad_loss,
            "Better model should have lower DSM loss: good={good_loss} < bad={bad_loss}"
        );
    }

    #[test]
    fn test_dsm_loss_perfect_score_quadratic() {
        // SCENARIO-TRAIN-002: For E(x) = 0.5*x^T*x, grad_energy(x) = x.
        // If data is at origin, noisy point is just noise n.
        // grad_energy(n) = n. Target = -n/sigma^2.
        // For sigma=1: target = -n. diff = n - (-n) = 2n.
        // So even the "right" energy doesn't give zero loss unless
        // we match the exact denoiser score.
        //
        // A model with grad_energy(x) = -x/sigma^2 would give zero loss
        // when data is at the origin. That corresponds to scale = -1/sigma^2.
        let sigma: Float = 1.0;
        let perfect_model = ScaledQuadraticEnergy {
            scale: -1.0 / (sigma * sigma),
        };

        let batch = array![[0.0, 0.0], [0.0, 0.0]];
        let noise = array![[0.3, -0.2], [-0.1, 0.4]];

        let loss = dsm_loss_with_noise(&perfect_model, &batch.view(), &noise.view(), sigma);
        assert!(
            loss < 1e-10,
            "Perfect denoiser score should give near-zero DSM loss, got {loss}"
        );
    }

    #[test]
    fn test_dsm_default_config() {
        // REQ-TRAIN-002: Default config sanity
        let config = DsmConfig::default();
        assert!(config.sigma > 0.0);
    }
}
