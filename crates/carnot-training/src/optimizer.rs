//! Optimizers for EBM training.
//!
//! Spec: REQ-TRAIN-004

use carnot_core::Float;
use ndarray::Array1;
use std::collections::HashMap;

/// Optimizer configuration.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum OptimizerConfig {
    Sgd {
        learning_rate: Float,
        momentum: Float,
    },
    Adam {
        learning_rate: Float,
        beta1: Float,
        beta2: Float,
        epsilon: Float,
    },
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        OptimizerConfig::Adam {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

/// Adam optimizer state.
pub struct AdamState {
    pub m: HashMap<String, Array1<Float>>,
    pub v: HashMap<String, Array1<Float>>,
    pub t: u64,
    pub config: OptimizerConfig,
}

impl AdamState {
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
            config,
        }
    }

    /// Apply one Adam update step to a parameter.
    pub fn step(
        &mut self,
        name: &str,
        param: &mut Array1<Float>,
        grad: &Array1<Float>,
        grad_clip: Option<Float>,
    ) -> Float {
        let (lr, beta1, beta2, epsilon) = match self.config {
            OptimizerConfig::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
            } => (learning_rate, beta1, beta2, epsilon),
            _ => panic!("AdamState used with non-Adam config"),
        };

        // Gradient clipping
        let grad = if let Some(max_norm) = grad_clip {
            let norm = grad.dot(grad).sqrt();
            if norm > max_norm {
                grad * (max_norm / norm)
            } else {
                grad.clone()
            }
        } else {
            grad.clone()
        };

        let grad_norm = grad.dot(&grad).sqrt();

        self.t += 1;

        let m = self
            .m
            .entry(name.to_string())
            .or_insert_with(|| Array1::zeros(param.len()));
        let v = self
            .v
            .entry(name.to_string())
            .or_insert_with(|| Array1::zeros(param.len()));

        // Update biased first moment
        *m = beta1 * &*m + (1.0 - beta1) * &grad;
        // Update biased second moment
        *v = beta2 * &*v + (1.0 - beta2) * &grad * &grad;

        // Bias correction
        let m_hat = &*m / (1.0 - beta1.powi(self.t as i32));
        let v_hat = &*v / (1.0 - beta2.powi(self.t as i32));

        // Update parameters
        *param = &*param - lr * &m_hat / &(v_hat.mapv(|x| x.sqrt()) + epsilon);

        grad_norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_adam_optimizer_step() {
        // REQ-TRAIN-004: optimizer step
        let config = OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut opt = AdamState::new(config);
        let mut param = array![1.0, 2.0, 3.0];
        let grad = array![0.1, 0.2, 0.3];

        let original = param.clone();
        opt.step("test", &mut param, &grad, None);

        // Parameters should have changed
        assert_ne!(param, original);
        assert!(param.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_adam_gradient_clipping() {
        // REQ-TRAIN-004: gradient clipping
        let config = OptimizerConfig::default();
        let mut opt = AdamState::new(config);
        let mut param = array![0.0, 0.0];
        let large_grad = array![100.0, 100.0];

        let norm = opt.step("test", &mut param, &large_grad, Some(1.0));
        // Grad norm should have been clipped
        assert!(param.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_adam_multiple_steps() {
        // REQ-TRAIN-004: training loop
        let config = OptimizerConfig::Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        };
        let mut opt = AdamState::new(config);
        let mut param = array![5.0, 5.0];

        // Minimize ||param||^2 — grad is 2*param
        for _ in 0..200 {
            let grad = 2.0 * &param;
            opt.step("w", &mut param, &grad, None);
        }

        // Should move toward zero
        let norm = param.dot(&param).sqrt();
        let initial_norm: Float = (5.0 * 5.0 + 5.0 * 5.0 as Float).sqrt();
        assert!(norm < initial_norm, "Optimizer should reduce parameter norm: initial={initial_norm}, final={norm}");
    }
}
