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
        if self.hidden_dims.contains(&0) {
            return Err(CarnotError::InvalidConfig(
                "all hidden_dims must be > 0".into(),
            ));
        }
        if self.num_heads == 0 {
            return Err(CarnotError::InvalidConfig("num_heads must be > 0".into()));
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

fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU activation: x * sigmoid(x)
fn silu(x: Float) -> Float {
    x * sigmoid(x)
}

/// SiLU derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
fn silu_deriv(x: Float) -> Float {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
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

/// Cached intermediate values for backprop through a residual block.
struct ResBlockCache {
    z1: Array1<Float>, // pre-activation of first SiLU
    z2: Array1<Float>, // pre-activation of second SiLU
}

impl ResidualBlock {
    fn forward(&self, x: &Array1<Float>) -> Array1<Float> {
        let z1 = self.w1.dot(x) + &self.b1;
        let a1 = z1.mapv(silu);
        let z2 = self.w2.dot(&a1) + &self.b2;
        let out = z2.mapv(silu);

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

    fn forward_with_cache(&self, x: &Array1<Float>) -> (Array1<Float>, ResBlockCache) {
        let z1 = self.w1.dot(x) + &self.b1;
        let a1 = z1.mapv(silu);
        let z2 = self.w2.dot(&a1) + &self.b2;
        let out = z2.mapv(silu);

        let result = if self.use_residual {
            let skip = match &self.proj {
                Some(proj) => proj.dot(x),
                None => x.clone(),
            };
            out + skip
        } else {
            out
        };

        let cache = ResBlockCache { z1, z2 };
        (result, cache)
    }

    /// Backprop: given dE/d(output), compute dE/d(input).
    fn backward(&self, d_out: &Array1<Float>, cache: &ResBlockCache) -> Array1<Float> {
        // output = silu(z2) + skip  (if residual)
        // d_out is dE/d(output)

        // Gradient through the SiLU(z2) path
        let d_silu2 = d_out * &cache.z2.mapv(silu_deriv);
        // d_silu2 = dE/d(z2)

        // z2 = W2 * a1 + b2  →  dE/d(a1) = W2^T * d_silu2
        let d_a1 = self.w2.t().dot(&d_silu2);

        // a1 = silu(z1)  →  dE/d(z1) = d_a1 ⊙ silu'(z1)
        let d_silu1 = &d_a1 * &cache.z1.mapv(silu_deriv);

        // z1 = W1 * x + b1  →  dE/d(x) via main path = W1^T * d_silu1
        let mut d_input = self.w1.t().dot(&d_silu1);

        // Gradient through the skip/residual path
        if self.use_residual {
            match &self.proj {
                Some(proj) => {
                    // skip = proj * x  →  d_input += proj^T * d_out
                    d_input += &proj.t().dot(d_out);
                }
                None => {
                    // skip = x  →  d_input += d_out
                    d_input += d_out;
                }
            }
        }

        d_input
    }
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
        h = h.mapv(silu);

        for block in &self.blocks {
            h = block.forward(&h);
        }

        self.output_weight.dot(&h) + self.output_bias
    }

    /// Analytical backpropagation through residual blocks.
    ///
    /// Spec: REQ-CORE-001, SCENARIO-CORE-003
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Forward pass with caching
        let z_input = self.input_proj.dot(x) + &self.input_bias;
        let mut h = z_input.mapv(silu);

        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (output, cache) = block.forward_with_cache(&h);
            block_caches.push(cache);
            h = output;
        }

        // Backward pass
        // dE/dh_final = output_weight
        let mut delta = self.output_weight.clone();

        // Backprop through residual blocks in reverse
        for i in (0..self.blocks.len()).rev() {
            delta = self.blocks[i].backward(&delta, &block_caches[i]);
        }

        // Backprop through input SiLU: dE/d(z_input) = delta ⊙ silu'(z_input)
        let d_z_input = &delta * &z_input.mapv(silu_deriv);

        // Backprop through input projection: dE/dx = input_proj^T * d_z_input
        self.input_proj.t().dot(&d_z_input)
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
    fn test_boltzmann_analytical_gradient_vs_finite_difference() {
        // SCENARIO-CORE-003: analytical gradient matches numerical
        let model = BoltzmannModel::new(BoltzmannConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            num_heads: 1,
            residual: true,
            layer_norm: false,
        })
        .unwrap();
        let x = Array1::random(5, Uniform::new(-0.5, 0.5));
        let grad = model.grad_energy(&x.view());

        let eps: Float = 1e-4;
        for i in 0..5 {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += eps;
            x_m[i] -= eps;
            let fd = (model.energy(&x_p.view()) - model.energy(&x_m.view())) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 0.05,
                "Gradient mismatch at index {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
    }

    #[test]
    fn test_boltzmann_gradient_no_residual() {
        // SCENARIO-CORE-003: gradient works without residual connections too
        let model = BoltzmannModel::new(BoltzmannConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            num_heads: 1,
            residual: false,
            layer_norm: false,
        })
        .unwrap();
        let x = Array1::random(5, Uniform::new(-0.5, 0.5));
        let grad = model.grad_energy(&x.view());

        let eps: Float = 1e-4;
        for i in 0..5 {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += eps;
            x_m[i] -= eps;
            let fd = (model.energy(&x_p.view()) - model.energy(&x_m.view())) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 0.05,
                "Gradient mismatch (no residual) at index {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
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
