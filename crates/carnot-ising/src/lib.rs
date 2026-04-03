//! carnot-ising: Ising (small tier) Energy Based Model.
//!
//! Implements a pairwise interaction energy model:
//!   E(x) = -0.5 * x^T J x - b^T x
//!
//! Spec: REQ-TIER-001, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006

use carnot_core::init::Initializer;
use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for the Ising model.
///
/// Spec: REQ-TIER-005
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IsingConfig {
    pub input_dim: usize,
    /// If None, direct pairwise interactions only.
    pub hidden_dim: Option<usize>,
    /// Initialization strategy name (xavier_uniform, he_normal, zeros).
    pub coupling_init: String,
}

impl IsingConfig {
    /// Validate configuration.
    ///
    /// Spec: SCENARIO-TIER-006
    pub fn validate(&self) -> Result<(), CarnotError> {
        if self.input_dim == 0 {
            return Err(CarnotError::InvalidConfig(
                "input_dim must be > 0".to_string(),
            ));
        }
        if let Some(h) = self.hidden_dim {
            if h == 0 {
                return Err(CarnotError::InvalidConfig(
                    "hidden_dim must be > 0 if specified".to_string(),
                ));
            }
        }
        Ok(())
    }
}

impl Default for IsingConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dim: None,
            coupling_init: "xavier_uniform".to_string(),
        }
    }
}

/// Ising Energy Based Model.
///
/// E(x) = -0.5 * x^T J x - b^T x
///
/// Spec: REQ-TIER-001
pub struct IsingModel {
    /// Coupling matrix J (symmetric).
    pub coupling: Array2<Float>,
    /// Bias vector b.
    pub bias: Array1<Float>,
    /// Configuration.
    pub config: IsingConfig,
}

impl IsingModel {
    /// Create a new Ising model with the given configuration.
    ///
    /// Spec: REQ-TIER-001, REQ-TIER-005, SCENARIO-TIER-006
    pub fn new(config: IsingConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        let init = match config.coupling_init.as_str() {
            "xavier_uniform" => Initializer::XavierUniform,
            "he_normal" => Initializer::HeNormal,
            "zeros" => Initializer::Zeros,
            other => {
                return Err(CarnotError::InvalidConfig(format!(
                    "Unknown initializer: {other}"
                )))
            }
        };

        let dim = config.input_dim;
        let mut coupling = init.init_matrix(dim, dim);
        // Make coupling symmetric: J = (J + J^T) / 2
        let coupling_t = coupling.t().to_owned();
        coupling = (&coupling + &coupling_t) / 2.0;

        let bias = Array1::zeros(dim);

        Ok(Self {
            coupling,
            bias,
            config,
        })
    }

    /// Memory footprint in bytes (parameters only).
    ///
    /// Spec: SCENARIO-TIER-005
    pub fn parameter_memory_bytes(&self) -> usize {
        let float_size = std::mem::size_of::<Float>();
        let coupling_size = self.coupling.len() * float_size;
        let bias_size = self.bias.len() * float_size;
        coupling_size + bias_size
    }
}

impl EnergyFunction for IsingModel {
    /// Spec: REQ-CORE-001, SCENARIO-CORE-001
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        -0.5 * x.dot(&self.coupling.dot(x)) - self.bias.dot(x)
    }

    /// Spec: REQ-CORE-001, SCENARIO-CORE-003
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // d/dx [-0.5 x^T J x - b^T x] = -J x - b  (since J is symmetric)
        let jx = self.coupling.dot(x);
        -&jx - &self.bias
    }

    fn input_dim(&self) -> usize {
        self.config.input_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_ising_default_config() {
        // REQ-TIER-005
        let config = IsingConfig::default();
        assert_eq!(config.input_dim, 784);
        assert!(config.hidden_dim.is_none());
    }

    #[test]
    fn test_ising_config_validation_zero_dim() {
        // SCENARIO-TIER-006
        let config = IsingConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ising_config_validation_zero_hidden() {
        // SCENARIO-TIER-006
        let config = IsingConfig {
            input_dim: 10,
            hidden_dim: Some(0),
            coupling_init: "xavier_uniform".to_string(),
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ising_creation() {
        // REQ-TIER-001
        let config = IsingConfig {
            input_dim: 10,
            hidden_dim: None,
            coupling_init: "xavier_uniform".to_string(),
        };
        let model = IsingModel::new(config).unwrap();
        assert_eq!(model.input_dim(), 10);
        assert_eq!(model.coupling.shape(), &[10, 10]);
        assert_eq!(model.bias.len(), 10);
    }

    #[test]
    fn test_ising_coupling_symmetric() {
        // REQ-TIER-001
        let config = IsingConfig {
            input_dim: 5,
            hidden_dim: None,
            coupling_init: "xavier_uniform".to_string(),
        };
        let model = IsingModel::new(config).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                assert!(
                    (model.coupling[[i, j]] - model.coupling[[j, i]]).abs() < 1e-10,
                    "Coupling matrix not symmetric at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn test_ising_energy_finite() {
        // SCENARIO-TIER-001, SCENARIO-CORE-001
        let model = IsingModel::new(IsingConfig {
            input_dim: 10,
            ..Default::default()
        })
        .unwrap();
        let x = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let e = model.energy(&x.view());
        assert!(e.is_finite(), "Energy should be finite, got {e}");
    }

    #[test]
    fn test_ising_energy_batch() {
        // SCENARIO-CORE-002
        let model = IsingModel::new(IsingConfig {
            input_dim: 3,
            ..Default::default()
        })
        .unwrap();
        let xs = ndarray::array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let energies = model.energy_batch(&xs.view());
        assert_eq!(energies.len(), 2);
        assert!(energies.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn test_ising_gradient_finite_difference() {
        // SCENARIO-CORE-003
        let model = IsingModel::new(IsingConfig {
            input_dim: 5,
            ..Default::default()
        })
        .unwrap();
        let x = array![0.5, -0.3, 0.8, -0.1, 0.4];
        let grad = model.grad_energy(&x.view());

        let eps = 1e-4 as Float;
        for i in 0..5 {
            let mut x_p = x.clone();
            let mut x_m = x.clone();
            x_p[i] += eps;
            x_m[i] -= eps;
            let fd = (model.energy(&x_p.view()) - model.energy(&x_m.view())) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 1e-3,
                "Gradient mismatch at index {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
    }

    #[test]
    fn test_ising_memory_footprint() {
        // SCENARIO-TIER-005: < 10MB for default config
        let model = IsingModel::new(IsingConfig::default()).unwrap();
        let bytes = model.parameter_memory_bytes();
        let mb = bytes as f64 / (1024.0 * 1024.0);
        assert!(
            mb < 10.0,
            "Memory footprint should be < 10MB, got {mb:.2}MB"
        );
    }

    #[test]
    fn test_ising_interface_conformance() {
        // SCENARIO-TIER-004: works through generic EnergyFunction trait
        fn compute_energy(model: &dyn EnergyFunction, x: &ArrayView1<Float>) -> Float {
            model.energy(x)
        }

        let model = IsingModel::new(IsingConfig {
            input_dim: 3,
            ..Default::default()
        })
        .unwrap();
        let x = array![1.0, 2.0, 3.0];
        let e = compute_energy(&model, &x.view());
        assert!(e.is_finite());
    }
}
