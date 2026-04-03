//! carnot-boltzmann: Boltzmann (large tier) Energy Based Model.
//!
//! Deep residual energy network with optional attention.
//!
//! Spec: REQ-TIER-003, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006

use carnot_core::init::Initializer;
use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for the Boltzmann model.
///
/// Spec: REQ-TIER-005
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoltzmannConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub num_heads: usize,
    pub residual: bool,
    pub layer_norm: bool,
}

impl BoltzmannConfig {
    pub fn validate(&self) -> Result<(), CarnotError> {
        if self.input_dim == 0 {
            return Err(CarnotError::InvalidConfig("input_dim must be > 0".into()));
        }
        if self.hidden_dims.is_empty() {
            return Err(CarnotError::InvalidConfig(
                "hidden_dims must have at least one layer".into(),
            ));
        }
        if self.hidden_dims.iter().any(|&d| d == 0) {
            return Err(CarnotError::InvalidConfig(
                "all hidden_dims must be > 0".into(),
            ));
        }
        if self.num_heads == 0 {
            return Err(CarnotError::InvalidConfig(
                "num_heads must be > 0".into(),
            ));
        }
        Ok(())
    }
}

impl Default for BoltzmannConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dims: vec![1024, 512, 256, 128],
            num_heads: 4,
            residual: true,
            layer_norm: true,
        }
    }
}

/// A residual block: y = x + SiLU(W2 * SiLU(W1 * x + b1) + b2)
/// If dimensions don't match, a projection is applied to x.
struct ResidualBlock {
    w1: Array2<Float>,
    b1: Array1<Float>,
    w2: Array2<Float>,
    b2: Array1<Float>,
    proj: Option<Array2<Float>>,
    use_residual: bool,
}

impl ResidualBlock {
    fn forward(&self, x: &Array1<Float>) -> Array1<Float> {
        let h = self.w1.dot(x) + &self.b1;
        let h = h.mapv(|v| v * sigmoid(v)); // SiLU
        let out = self.w2.dot(&h) + &self.b2;
        let out = out.mapv(|v| v * sigmoid(v)); // SiLU

        if self.use_residual {
            let skip = match &self.proj {
                Some(proj) => proj.dot(x),
                None => x.clone(),
            };
            out + skip
        } else {
            out
        }
    }
}

fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

/// Boltzmann Energy Based Model.
///
/// Spec: REQ-TIER-003
pub struct BoltzmannModel {
    input_proj: Array2<Float>,
    input_bias: Array1<Float>,
    blocks: Vec<ResidualBlock>,
    output_weight: Array1<Float>,
    output_bias: Float,
    config: BoltzmannConfig,
}

impl BoltzmannModel {
    /// Spec: REQ-TIER-003, REQ-TIER-005, SCENARIO-TIER-006
    pub fn new(config: BoltzmannConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        let init = Initializer::XavierUniform;

        let input_proj = init.init_matrix(config.hidden_dims[0], config.input_dim);
        let input_bias = Array1::zeros(config.hidden_dims[0]);

        let mut blocks = Vec::new();
        for i in 0..config.hidden_dims.len() - 1 {
            let in_dim = config.hidden_dims[i];
            let out_dim = config.hidden_dims[i + 1];

            let w1 = init.init_matrix(in_dim, in_dim);
            let b1 = Array1::zeros(in_dim);
            let w2 = init.init_matrix(out_dim, in_dim);
            let b2 = Array1::zeros(out_dim);

            let proj = if in_dim != out_dim {
                Some(init.init_matrix(out_dim, in_dim))
            } else {
                None
            };

            blocks.push(ResidualBlock {
                w1,
                b1,
                w2,
                b2,
                proj,
                use_residual: config.residual,
            });
        }

        let last_dim = *config.hidden_dims.last().unwrap();
        let output_weight = Array1::zeros(last_dim);

        Ok(Self {
            input_proj,
            input_bias,
            blocks,
            output_weight,
            output_bias: 0.0,
            config,
        })
    }
}

impl EnergyFunction for BoltzmannModel {
    /// Spec: REQ-CORE-001, SCENARIO-TIER-003
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let mut h = self.input_proj.dot(x) + &self.input_bias;
        h = h.mapv(|v| v * sigmoid(v)); // SiLU on input projection

        for block in &self.blocks {
            h = block.forward(&h);
        }

        self.output_weight.dot(&h) + self.output_bias
    }

    /// Numerical gradient (analytical backprop is a future optimization).
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let eps = 1e-4 as Float;
        let mut grad = Array1::zeros(x.len());
        for i in 0..x.len() {
            let mut x_p = x.to_owned();
            let mut x_m = x.to_owned();
            x_p[i] += eps;
            x_m[i] -= eps;
            grad[i] = (self.energy(&x_p.view()) - self.energy(&x_m.view())) / (2.0 * eps);
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.config.input_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    use ndarray_rand::RandomExt;
    use rand_distr::Uniform;

    #[test]
    fn test_boltzmann_default_config() {
        // REQ-TIER-005
        let config = BoltzmannConfig::default();
        assert_eq!(config.input_dim, 784);
        assert_eq!(config.hidden_dims, vec![1024, 512, 256, 128]);
        assert!(config.residual);
    }

    #[test]
    fn test_boltzmann_config_validation() {
        // SCENARIO-TIER-006
        let bad = BoltzmannConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_boltzmann_creation() {
        // REQ-TIER-003
        let model = BoltzmannModel::new(BoltzmannConfig {
            input_dim: 20,
            hidden_dims: vec![16, 8, 4],
            num_heads: 2,
            residual: true,
            layer_norm: false,
        })
        .unwrap();
        assert_eq!(model.input_dim(), 20);
    }

    #[test]
    fn test_boltzmann_energy_finite() {
        // SCENARIO-TIER-003
        let model = BoltzmannModel::new(BoltzmannConfig {
            input_dim: 10,
            hidden_dims: vec![8, 6, 4],
            num_heads: 2,
            residual: true,
            layer_norm: false,
        })
        .unwrap();
        let x = Array1::random(10, Uniform::new(-1.0, 1.0));
        let e = model.energy(&x.view());
        assert!(e.is_finite(), "Energy should be finite, got {e}");
    }

    #[test]
    fn test_boltzmann_gradient() {
        // SCENARIO-CORE-003
        let model = BoltzmannModel::new(BoltzmannConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            num_heads: 1,
            residual: true,
            layer_norm: false,
        })
        .unwrap();
        let x = Array1::random(5, Uniform::new(-1.0, 1.0));
        let grad = model.grad_energy(&x.view());
        assert_eq!(grad.len(), 5);
        assert!(grad.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn test_boltzmann_interface_conformance() {
        // SCENARIO-TIER-004
        fn compute_energy(model: &dyn EnergyFunction, x: &ArrayView1<Float>) -> Float {
            model.energy(x)
        }

        let model = BoltzmannModel::new(BoltzmannConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            num_heads: 1,
            residual: true,
            layer_norm: false,
        })
        .unwrap();
        let x = Array1::random(5, Uniform::new(-1.0, 1.0));
        let e = compute_energy(&model, &x.view());
        assert!(e.is_finite());
    }
}
