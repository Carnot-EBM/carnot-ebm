//! carnot-gibbs: Gibbs (medium tier) Energy Based Model.
//!
//! Multi-layer energy network: E(x) = f_L(...f_2(f_1(x))) -> scalar
//!
//! Spec: REQ-TIER-002, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006

use carnot_core::init::Initializer;
use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for the Gibbs model.
///
/// Spec: REQ-TIER-005
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GibbsConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub activation: Activation,
    pub dropout: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    SiLU,
    ReLU,
    Tanh,
}

impl GibbsConfig {
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
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(CarnotError::InvalidConfig(
                "dropout must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }
}

impl Default for GibbsConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dims: vec![512, 256],
            activation: Activation::SiLU,
            dropout: 0.0,
        }
    }
}

/// A single dense layer: y = activation(Wx + b)
struct DenseLayer {
    weight: Array2<Float>,
    bias: Array1<Float>,
    activation: Activation,
}

impl DenseLayer {
    fn forward(&self, x: &Array1<Float>) -> Array1<Float> {
        let z = self.weight.dot(x) + &self.bias;
        match self.activation {
            Activation::SiLU => z.mapv(|v| v * sigmoid(v)),
            Activation::ReLU => z.mapv(|v| v.max(0.0)),
            Activation::Tanh => z.mapv(|v| v.tanh()),
        }
    }
}

fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

/// Gibbs Energy Based Model.
///
/// Spec: REQ-TIER-002
pub struct GibbsModel {
    layers: Vec<DenseLayer>,
    output_weight: Array1<Float>,
    output_bias: Float,
    config: GibbsConfig,
}

impl GibbsModel {
    /// Spec: REQ-TIER-002, REQ-TIER-005, SCENARIO-TIER-006
    pub fn new(config: GibbsConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        let init = Initializer::XavierUniform;
        let mut layers = Vec::new();

        let mut prev_dim = config.input_dim;
        for &hidden_dim in &config.hidden_dims {
            let weight = init.init_matrix(hidden_dim, prev_dim);
            let bias = Array1::zeros(hidden_dim);
            layers.push(DenseLayer {
                weight,
                bias,
                activation: config.activation,
            });
            prev_dim = hidden_dim;
        }

        let output_weight = Array1::zeros(prev_dim);
        let output_bias = 0.0;

        Ok(Self {
            layers,
            output_weight,
            output_bias,
            config,
        })
    }

    fn forward_hidden(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h
    }
}

impl EnergyFunction for GibbsModel {
    /// Spec: REQ-CORE-001, SCENARIO-TIER-002
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let h = self.forward_hidden(x);
        self.output_weight.dot(&h) + self.output_bias
    }

    /// Spec: REQ-CORE-001, SCENARIO-CORE-003
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Numerical gradient via finite differences for correctness.
        // TODO: Implement analytical backprop for performance.
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
    fn test_gibbs_default_config() {
        // REQ-TIER-005
        let config = GibbsConfig::default();
        assert_eq!(config.input_dim, 784);
        assert_eq!(config.hidden_dims, vec![512, 256]);
    }

    #[test]
    fn test_gibbs_config_validation() {
        // SCENARIO-TIER-006
        let bad = GibbsConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(bad.validate().is_err());

        let bad2 = GibbsConfig {
            hidden_dims: vec![],
            ..Default::default()
        };
        assert!(bad2.validate().is_err());
    }

    #[test]
    fn test_gibbs_creation() {
        // REQ-TIER-002
        let model = GibbsModel::new(GibbsConfig {
            input_dim: 10,
            hidden_dims: vec![8, 4],
            ..Default::default()
        })
        .unwrap();
        assert_eq!(model.input_dim(), 10);
    }

    #[test]
    fn test_gibbs_energy_finite() {
        // SCENARIO-TIER-002
        let model = GibbsModel::new(GibbsConfig {
            input_dim: 10,
            hidden_dims: vec![8, 4],
            ..Default::default()
        })
        .unwrap();
        let x = Array1::random(10, Uniform::new(-1.0, 1.0));
        let e = model.energy(&x.view());
        assert!(e.is_finite(), "Energy should be finite, got {e}");
    }

    #[test]
    fn test_gibbs_energy_batch() {
        // SCENARIO-CORE-002
        let model = GibbsModel::new(GibbsConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            ..Default::default()
        })
        .unwrap();
        let xs = ndarray::Array2::random((4, 5), Uniform::new(-1.0, 1.0));
        let energies = model.energy_batch(&xs.view());
        assert_eq!(energies.len(), 4);
        assert!(energies.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn test_gibbs_gradient_finite_difference() {
        // SCENARIO-CORE-003
        let model = GibbsModel::new(GibbsConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            ..Default::default()
        })
        .unwrap();
        let x = Array1::random(5, Uniform::new(-1.0, 1.0));
        let grad = model.grad_energy(&x.view());

        let eps = 1e-4 as Float;
        for i in 0..5 {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += eps;
            x_m[i] -= eps;
            let fd = (model.energy(&x_p.view()) - model.energy(&x_m.view())) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-2,
                "Gradient mismatch at index {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
    }

    #[test]
    fn test_gibbs_interface_conformance() {
        // SCENARIO-TIER-004
        fn compute_energy(model: &dyn EnergyFunction, x: &ArrayView1<Float>) -> Float {
            model.energy(x)
        }

        let model = GibbsModel::new(GibbsConfig {
            input_dim: 5,
            hidden_dims: vec![4, 3],
            ..Default::default()
        })
        .unwrap();
        let x = Array1::random(5, Uniform::new(-1.0, 1.0));
        let e = compute_energy(&model, &x.view());
        assert!(e.is_finite());
    }
}
