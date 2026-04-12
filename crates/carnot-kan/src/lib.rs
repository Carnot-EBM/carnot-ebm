//! # carnot-kan: KAN (Kolmogorov-Arnold Networks) Energy-Based Model
//!
//! ## For Researchers
//!
//! KAN (Kolmogorov-Arnold Networks) addresses the constraint learning ceiling
//! where linear Ising features cannot capture nonlinear constraint relationships.
//!
//! Energy function:
//!   E(x) = sum_ij f_ij(s_i * s_j) + sum_i g_i(s_i)
//!
//! where f_ij and g_i are learnable B-spline functions.
//!
//! Spec: REQ-CORE-001, REQ-TIER-005, REQ-TIER-006

use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, ArrayView1};

/// BSpline parameters: control points at knot positions.
///
/// TODO: Implement proper B-spline basis function evaluation
/// Currently this is a placeholder for the spline coefficients.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BSplineParams {
    /// Control point values at knot positions.
    /// Shape: (num_knots + degree,) for degree-d spline with num_knots intervals.
    pub control_points: Vec<Float>,
}

/// B-spline evaluation at a point using lookup table.
///
/// TODO: Implement proper B-spline basis function evaluation
/// This uses simple linear interpolation between knot points.
/// For FPGA path, we want a lookup table approach.
#[derive(Debug, Clone)]
pub struct BSpline {
    /// Number of knots in the spline.
    pub num_knots: usize,
    /// Polynomial degree (0 = piecewise constant, 1 = linear, 3 = cubic).
    pub degree: usize,
    /// Knot positions.
    pub knot_positions: Vec<Float>,
    /// Control point values.
    pub control_points: Vec<Float>,
}

impl BSpline {
    /// Create a new B-spline with random initialization.
    ///
    /// TODO: Implement proper initialization
    pub fn new(num_knots: usize, degree: usize) -> Self {
        let n_params = num_knots + degree;
        let control_points = vec![0.0; n_params];
        let knot_positions = Self::compute_knot_positions(num_knots, degree);

        Self {
            num_knots,
            degree,
            knot_positions,
            control_points,
        }
    }

    /// Compute uniform knot positions with extension for boundary conditions.
    fn compute_knot_positions(num_knots: usize, degree: usize) -> Vec<Float> {
        let interior = if num_knots >= 2 { num_knots } else { 2 };
        let mut knots: Vec<Float> = (0..interior)
            .map(|i| -1.0 + 2.0 * i as Float / (interior - 1) as Float)
            .collect();

        for _ in 0..degree {
            knots.insert(0, knots[0] - 1.0);
            knots.push(knots[knots.len() - 1] + 1.0);
        }

        knots
    }

    /// Evaluate the spline at a single point.
    ///
    /// TODO: Implement proper B-spline evaluation with basis functions
    /// Currently uses simple linear interpolation.
    pub fn evaluate(&self, x: Float) -> Float {
        let x_clamped = x
            .max(self.knot_positions[0])
            .min(self.knot_positions[self.knot_positions.len() - 1]);
        let t = (x_clamped - self.knot_positions[0])
            / (self.knot_positions[self.knot_positions.len() - 1] - self.knot_positions[0]);

        let n = self.control_points.len();
        let idx = (t * (n - 1) as Float).floor() as usize;
        let idx = idx.min(n - 2);

        let t_local = t * (n - 1) as Float - idx as Float;

        self.control_points[idx] * (1.0 - t_local) + self.control_points[idx + 1] * t_local
    }

    /// Evaluate the spline at multiple points.
    ///
    /// TODO: Implement vectorized evaluation
    pub fn evaluate_batch(&self, xs: &[Float]) -> Vec<Float> {
        xs.iter().map(|&x| self.evaluate(x)).collect()
    }
}

/// KAN configuration.
///
/// TODO: Add full specification
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct KANConfig {
    /// Number of input dimensions.
    pub input_dim: usize,
    /// Number of knots per spline.
    pub num_knots: usize,
    /// Spline polynomial degree.
    pub degree: usize,
    /// Whether to use sparse connectivity.
    pub sparse: bool,
    /// Edge density if sparse (fraction of edges to keep).
    pub edge_density: Float,
}

impl Default for KANConfig {
    fn default() -> Self {
        Self {
            input_dim: 784,
            num_knots: 10,
            degree: 3,
            sparse: true,
            edge_density: 0.1,
        }
    }
}

impl KANConfig {
    /// Validate configuration.
    pub fn validate(&self) -> Result<(), CarnotError> {
        if self.input_dim == 0 {
            return Err(CarnotError::InvalidConfig(
                "input_dim must be > 0".to_string(),
            ));
        }
        if self.num_knots < 2 {
            return Err(CarnotError::InvalidConfig(
                "num_knots must be >= 2".to_string(),
            ));
        }
        if self.edge_density <= 0.0 || self.edge_density > 1.0 {
            return Err(CarnotError::InvalidConfig(
                "edge_density must be in (0, 1]".to_string(),
            ));
        }
        Ok(())
    }
}

/// KAN Energy Function with learnable B-spline edges and biases.
///
/// Energy: E(x) = sum_ij f_ij(s_i * s_j) + sum_i g_i(s_i)
///
/// TODO: Implement full KAN energy computation
#[derive(Debug, Clone)]
pub struct KANEnergyFunction {
    /// Input dimension.
    input_dim: usize,
    /// Edge splines: mapping from (i, j) to BSpline.
    #[allow(dead_code)]
    edge_splines: Vec<((usize, usize), BSpline)>,
    /// Bias splines: one per node.
    bias_splines: Vec<BSpline>,
    /// Edge list for iteration.
    #[allow(dead_code)]
    edges: Vec<(usize, usize)>,
}

impl KANEnergyFunction {
    /// Create a new KAN energy function.
    ///
    /// TODO: Implement full initialization with edge selection
    pub fn new(config: KANConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        let edge_splines: Vec<((usize, usize), BSpline)> = Vec::new();
        let bias_splines: Vec<BSpline> = (0..config.input_dim)
            .map(|_| BSpline::new(config.num_knots, config.degree))
            .collect();

        let edges: Vec<(usize, usize)> = Vec::new();

        Ok(Self {
            input_dim: config.input_dim,
            edge_splines,
            bias_splines,
            edges,
        })
    }

    /// Compute the energy for a single input.
    ///
    /// TODO: Implement full energy computation with spline evaluation
    pub fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let mut result = 0.0;

        for spline in &self.bias_splines {
            for &xi in x {
                result += spline.evaluate(xi);
            }
        }

        result
    }

    /// Get the gradient of the energy with respect to input.
    ///
    /// TODO: Implement gradient computation through splines
    #[allow(dead_code)]
    pub fn grad_energy(&self, _x: &ArrayView1<Float>) -> Array1<Float> {
        Array1::zeros(self.input_dim)
    }

    /// Get the number of learnable parameters.
    pub fn n_params(&self) -> usize {
        let edge_params: usize = self
            .edge_splines
            .iter()
            .map(|(_, s)| s.control_points.len())
            .sum();
        let bias_params: usize = self
            .bias_splines
            .iter()
            .map(|s| s.control_points.len())
            .sum();
        edge_params + bias_params
    }
}

impl EnergyFunction for KANEnergyFunction {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        KANEnergyFunction::energy(self, x)
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        KANEnergyFunction::grad_energy(self, x)
    }

    fn input_dim(&self) -> usize {
        self.input_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // REQ-CORE-001: BSpline constructor produces a valid finite lookup object
    #[test]
    fn test_bspline_creation() {
        let spline = BSpline::new(10, 3);
        assert_eq!(spline.num_knots, 10);
        assert_eq!(spline.degree, 3);
    }

    // REQ-CORE-001, SCENARIO-CORE-001: BSpline evaluation returns a finite scalar
    #[test]
    fn test_bspline_evaluate() {
        let spline = BSpline::new(5, 1);
        let result = spline.evaluate(0.0);
        assert!(result.is_finite());
    }

    // REQ-TIER-005: default KAN configuration exposes sensible defaults
    #[test]
    fn test_kan_config_default() {
        let config = KANConfig::default();
        assert_eq!(config.input_dim, 784);
        assert_eq!(config.num_knots, 10);
        assert_eq!(config.degree, 3);
    }

    // REQ-TIER-005, SCENARIO-TIER-006: invalid config is rejected
    #[test]
    fn test_kan_config_validation() {
        let config = KANConfig {
            input_dim: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // REQ-CORE-001, REQ-TIER-005: KAN energy function constructs from config
    #[test]
    fn test_kan_energy_function_creation() {
        let config = KANConfig::default();
        let kan = KANEnergyFunction::new(config).unwrap();
        assert_eq!(kan.input_dim(), 784);
    }

    // REQ-CORE-001, SCENARIO-CORE-001: KAN energy is finite for valid input
    #[test]
    fn test_kan_energy_finite() {
        let config = KANConfig {
            input_dim: 10,
            ..Default::default()
        };
        let kan = KANEnergyFunction::new(config).unwrap();
        let x = Array1::from_vec(vec![0.0; 10]);
        let e = kan.energy(&x.view());
        assert!(e.is_finite());
    }

    // REQ-CORE-001, SCENARIO-CORE-003: gradient shape matches input dimension
    #[test]
    fn test_kan_gradient_shape() {
        let config = KANConfig {
            input_dim: 10,
            ..Default::default()
        };
        let kan = KANEnergyFunction::new(config).unwrap();
        let x = Array1::from_vec(vec![0.0; 10]);
        let grad = kan.grad_energy(&x.view());
        assert_eq!(grad.len(), 10);
    }

    // REQ-TIER-005, REQ-TIER-006: parameter count is positive for valid config
    #[test]
    fn test_kan_n_params() {
        let config = KANConfig {
            input_dim: 10,
            num_knots: 5,
            degree: 2,
            sparse: false,
            edge_density: 1.0,
        };
        let kan = KANEnergyFunction::new(config).unwrap();
        let n = kan.n_params();
        assert!(n > 0);
    }
}
