use ndarray::{ArrayView2, Axis};
use carnot_core::Float;

/// Computes the Energy-Based Fine-Tuning (EBFT) objective.
pub fn ebft_loss_contrastive(
    expert_energies: &[Float],
    rollout_energies: &[Float],
) -> Float {
    let expert_mean = expert_energies.iter().copied().sum::<Float>() / expert_energies.len() as Float;
    let rollout_mean = rollout_energies.iter().copied().sum::<Float>() / rollout_energies.len() as Float;
    expert_mean - rollout_mean
}

/// Computes the EBFT feature-matching objective.
pub fn ebft_loss_features(
    model_features: ArrayView2<Float>,
    target_features: ArrayView2<Float>,
) -> Float {
    let model_expected = model_features.mean_axis(Axis(0)).unwrap();
    let target_expected = target_features.mean_axis(Axis(0)).unwrap();
    let diff = &model_expected - &target_expected;
    diff.mapv(|v| v * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ebft_loss_contrastive() {
        let expert = vec![1.0, 2.0];
        let rollout = vec![3.0, 4.0];
        let loss = ebft_loss_contrastive(&expert, &rollout);
        assert!((loss - (1.5 - 3.5)).abs() < 1e-6);
    }

    #[test]
    fn test_ebft_loss_features() {
        let model = array![[1.0, 2.0], [3.0, 4.0]]; // mean: [2.0, 3.0]
        let target = array![[0.0, 1.0], [2.0, 3.0]]; // mean: [1.0, 2.0]
        let loss = ebft_loss_features(model.view(), target.view());
        // diff: [1.0, 1.0], squared sum: 2.0
        assert!((loss - 2.0).abs() < 1e-6);
    }
}

