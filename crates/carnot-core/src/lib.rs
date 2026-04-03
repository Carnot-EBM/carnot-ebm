//! carnot-core: Core EBM traits, types, and abstractions.
//!
//! Implements: REQ-CORE-001, REQ-CORE-003, REQ-CORE-004, REQ-CORE-006

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::collections::HashMap;

pub mod error;
pub mod init;
pub mod serialize;

pub use error::CarnotError;

// REQ-CORE-006: Numeric precision
#[cfg(not(feature = "f64"))]
pub type Float = f32;
#[cfg(feature = "f64")]
pub type Float = f64;

/// Core trait for energy-based models.
/// All model tiers implement this trait.
///
/// Spec: REQ-CORE-001
pub trait EnergyFunction: Send + Sync {
    /// Compute scalar energy for a single input.
    fn energy(&self, x: &ArrayView1<Float>) -> Float;

    /// Compute energy for a batch of inputs.
    /// Default implementation maps over batch dimension.
    fn energy_batch(&self, xs: &ArrayView2<Float>) -> Array1<Float> {
        Array1::from_iter(xs.rows().into_iter().map(|row| self.energy(&row)))
    }

    /// Compute gradient of energy w.r.t. input.
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float>;

    /// Number of input dimensions this model expects.
    fn input_dim(&self) -> usize;
}

/// Model configuration.
///
/// Spec: REQ-CORE-003
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
    pub precision: Precision,
}

/// Numeric precision selection.
///
/// Spec: REQ-CORE-006
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Precision {
    F32,
    F64,
}

impl Default for Precision {
    fn default() -> Self {
        Precision::F32
    }
}

/// Model training metadata.
///
/// Spec: REQ-CORE-003
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    pub step: u64,
    pub loss_history: Vec<Float>,
}

/// Complete model state for serialization.
///
/// Spec: REQ-CORE-003
#[derive(Debug, Clone)]
pub struct ModelState {
    pub parameters: HashMap<String, Array1<Float>>,
    pub config: ModelConfig,
    pub metadata: ModelMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A trivial energy function for testing the trait.
    /// E(x) = 0.5 * ||x||^2
    struct QuadraticEnergy;

    impl EnergyFunction for QuadraticEnergy {
        // SCENARIO-CORE-001: Compute Energy for Single Input
        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            0.5 * x.dot(x)
        }

        // SCENARIO-CORE-003: Compute Energy Gradient
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            x.to_owned()
        }

        fn input_dim(&self) -> usize {
            0 // accepts any dimension
        }
    }

    #[test]
    fn test_quadratic_energy_single() {
        // SCENARIO-CORE-001
        let model = QuadraticEnergy;
        let x = array![1.0, 2.0, 3.0];
        let e = model.energy(&x.view());
        assert!((e - 7.0).abs() < 1e-6);
        assert!(e.is_finite());
    }

    #[test]
    fn test_quadratic_energy_batch() {
        // SCENARIO-CORE-002
        let model = QuadraticEnergy;
        let xs = ndarray::array![[1.0, 0.0], [0.0, 2.0]];
        let energies = model.energy_batch(&xs.view());
        assert_eq!(energies.len(), 2);
        assert!((energies[0] - 0.5).abs() < 1e-6);
        assert!((energies[1] - 2.0).abs() < 1e-6);
        assert!(energies.iter().all(|e| e.is_finite()));
    }

    #[test]
    fn test_quadratic_gradient() {
        // SCENARIO-CORE-003
        let model = QuadraticEnergy;
        let x = array![1.0, 2.0, 3.0];
        let grad = model.grad_energy(&x.view());
        assert_eq!(grad.len(), x.len());

        // Verify gradient matches finite difference
        let eps: Float = 1e-3;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let e_plus = model.energy(&x_plus.view());
            let e_minus = model.energy(&x_minus.view());
            let fd = (e_plus - e_minus) / (2.0 * eps);
            assert!((grad[i] - fd).abs() < 0.1, "Gradient mismatch at index {i}: analytic={}, fd={fd}", grad[i]);
        }
    }

    #[test]
    fn test_precision_default() {
        // REQ-CORE-006
        assert_eq!(Precision::default(), Precision::F32);
    }
}
