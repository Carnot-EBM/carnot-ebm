//! Analytical benchmark energy functions with known global minima.
//!
//! These serve as evaluation targets for the autoresearch loop (REQ-AUTO-001)
//! and as E2E verification of the sampler/training pipeline.
//!
//! Each benchmark defines:
//! - Energy function (implements EnergyFunction)
//! - Known global minimum location and energy
//! - Evaluation metrics
//!
//! Spec: REQ-AUTO-001

pub mod runner;

use ndarray::{Array1, ArrayView1};
use std::f32::consts::PI;

use crate::{EnergyFunction, Float};

/// Metadata about a benchmark's known optimal solution.
#[derive(Debug, Clone)]
pub struct BenchmarkInfo {
    pub name: &'static str,
    pub input_dim: usize,
    pub global_min_energy: Float,
    pub global_min_location: Array1<Float>,
    pub description: &'static str,
}

/// Double-well potential: E(x) = (x[0]^2 - 1)^2 + sum(x[1:]^2)
///
/// Two symmetric global minima at x = [±1, 0, 0, ...] with energy 0.
/// Tests ability to find multimodal minima.
///
/// Spec: REQ-AUTO-001
pub struct DoubleWell {
    dim: usize,
}

impl DoubleWell {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "DoubleWell requires dim >= 1");
        Self { dim }
    }

    pub fn info(&self) -> BenchmarkInfo {
        let mut loc = Array1::zeros(self.dim);
        loc[0] = 1.0;
        BenchmarkInfo {
            name: "double_well",
            input_dim: self.dim,
            global_min_energy: 0.0,
            global_min_location: loc,
            description: "E(x) = (x[0]^2 - 1)^2 + sum(x[1:]^2). Two minima at x[0]=±1.",
        }
    }
}

impl EnergyFunction for DoubleWell {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let well = (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0);
        let rest: Float = x.iter().skip(1).map(|&xi| xi * xi).sum();
        well + rest
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        // d/dx0 [(x0^2 - 1)^2] = 4*x0*(x0^2 - 1)
        grad[0] = 4.0 * x[0] * (x[0] * x[0] - 1.0);
        // d/dxi [xi^2] = 2*xi for i > 0
        for i in 1..x.len() {
            grad[i] = 2.0 * x[i];
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Rosenbrock function: E(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]
///
/// Global minimum at x = [1, 1, ..., 1] with energy 0.
/// Famous for its narrow curved valley — hard optimization test.
///
/// Spec: REQ-AUTO-001
pub struct Rosenbrock {
    dim: usize,
}

impl Rosenbrock {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2, "Rosenbrock requires dim >= 2");
        Self { dim }
    }

    pub fn info(&self) -> BenchmarkInfo {
        BenchmarkInfo {
            name: "rosenbrock",
            input_dim: self.dim,
            global_min_energy: 0.0,
            global_min_location: Array1::ones(self.dim),
            description: "Rosenbrock banana function. Min at [1,1,...,1].",
        }
    }
}

impl EnergyFunction for Rosenbrock {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let mut sum: Float = 0.0;
        for i in 0..(x.len() - 1) {
            let a = x[i + 1] - x[i] * x[i];
            let b = 1.0 - x[i];
            sum += 100.0 * a * a + b * b;
        }
        sum
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let n = x.len();
        let mut grad = Array1::zeros(n);
        for i in 0..(n - 1) {
            let a = x[i + 1] - x[i] * x[i];
            // d/dx[i]: -400*x[i]*(x[i+1] - x[i]^2) + 2*(x[i] - 1)
            grad[i] += -400.0 * x[i] * a + 2.0 * (x[i] - 1.0);
            // d/dx[i+1]: 200*(x[i+1] - x[i]^2)
            grad[i + 1] += 200.0 * a;
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Ackley function: a multimodal function with many local minima.
///
/// E(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e
///
/// Global minimum at x = [0, 0, ..., 0] with energy ≈ 0.
/// Tests ability to escape local minima.
///
/// Spec: REQ-AUTO-001
pub struct Ackley {
    dim: usize,
}

impl Ackley {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "Ackley requires dim >= 1");
        Self { dim }
    }

    pub fn info(&self) -> BenchmarkInfo {
        BenchmarkInfo {
            name: "ackley",
            input_dim: self.dim,
            global_min_energy: 0.0,
            global_min_location: Array1::zeros(self.dim),
            description: "Ackley multimodal function. Min at origin.",
        }
    }
}

impl EnergyFunction for Ackley {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let n = x.len() as Float;
        let sum_sq: Float = x.iter().map(|&xi| xi * xi).sum();
        let sum_cos: Float = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum();
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f32::consts::E
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Numerical gradient for Ackley (analytical is messy)
        let eps: Float = 1e-4;
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
        self.dim
    }
}

/// Rastrigin function: highly multimodal with regular local minima.
///
/// E(x) = 10*n + sum(x[i]^2 - 10*cos(2*pi*x[i]))
///
/// Global minimum at x = [0, 0, ..., 0] with energy 0.
///
/// Spec: REQ-AUTO-001
pub struct Rastrigin {
    dim: usize,
}

impl Rastrigin {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "Rastrigin requires dim >= 1");
        Self { dim }
    }

    pub fn info(&self) -> BenchmarkInfo {
        BenchmarkInfo {
            name: "rastrigin",
            input_dim: self.dim,
            global_min_energy: 0.0,
            global_min_location: Array1::zeros(self.dim),
            description: "Rastrigin function with many regular local minima. Min at origin.",
        }
    }
}

impl EnergyFunction for Rastrigin {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let n = x.len() as Float;
        let sum: Float = x
            .iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum();
        10.0 * n + sum
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        Array1::from_iter(
            x.iter()
                .map(|&xi| 2.0 * xi + 10.0 * 2.0 * PI * (2.0 * PI * xi).sin()),
        )
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Gaussian mixture energy: -log(sum_k w_k * N(x; mu_k, sigma_k))
///
/// Known parameters make this ideal for testing whether sampling
/// recovers the correct distribution.
///
/// Spec: REQ-AUTO-001
pub struct GaussianMixture {
    dim: usize,
    means: Vec<Array1<Float>>,
    variances: Vec<Float>,
    weights: Vec<Float>,
}

impl GaussianMixture {
    /// Create a simple 1D mixture of two Gaussians.
    pub fn two_modes(separation: Float) -> Self {
        Self {
            dim: 1,
            means: vec![
                Array1::from_elem(1, -separation / 2.0),
                Array1::from_elem(1, separation / 2.0),
            ],
            variances: vec![1.0, 1.0],
            weights: vec![0.5, 0.5],
        }
    }

    pub fn info(&self) -> BenchmarkInfo {
        BenchmarkInfo {
            name: "gaussian_mixture",
            input_dim: self.dim,
            global_min_energy: 0.0, // approximate
            global_min_location: self.means[0].clone(),
            description: "Gaussian mixture model with known parameters.",
        }
    }
}

impl EnergyFunction for GaussianMixture {
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        let mut log_sum: Float = Float::NEG_INFINITY;
        for (mean, (&var, &w)) in self
            .means
            .iter()
            .zip(self.variances.iter().zip(self.weights.iter()))
        {
            let diff = x.to_owned() - mean;
            let exponent = -0.5 * diff.dot(&diff) / var;
            let log_component =
                w.ln() + exponent - 0.5 * (self.dim as Float) * (2.0 * PI * var).ln();
            // log-sum-exp for numerical stability
            if log_sum == Float::NEG_INFINITY {
                log_sum = log_component;
            } else {
                let max = log_sum.max(log_component);
                log_sum = max + ((log_sum - max).exp() + (log_component - max).exp()).ln();
            }
        }
        -log_sum // negative log-likelihood = energy
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Numerical gradient
        let eps: Float = 1e-4;
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
        self.dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_double_well_at_minimum() {
        // REQ-AUTO-001: DoubleWell energy = 0 at minimum
        let dw = DoubleWell::new(3);
        let info = dw.info();
        let e = dw.energy(&info.global_min_location.view());
        assert!(e.abs() < 1e-6, "Energy at minimum should be ~0, got {e}");
    }

    #[test]
    fn test_double_well_symmetry() {
        // REQ-AUTO-001: two symmetric minima
        let dw = DoubleWell::new(2);
        let x_pos = array![1.0, 0.0];
        let x_neg = array![-1.0, 0.0];
        let e_pos = dw.energy(&x_pos.view());
        let e_neg = dw.energy(&x_neg.view());
        assert!((e_pos - e_neg).abs() < 1e-6);
        assert!(e_pos < 1e-6);
    }

    #[test]
    fn test_double_well_gradient() {
        // REQ-AUTO-001: gradient at minimum is zero
        let dw = DoubleWell::new(2);
        let x = array![1.0, 0.0];
        let grad = dw.grad_energy(&x.view());
        assert!(grad.iter().all(|g| g.abs() < 1e-5));
    }

    #[test]
    fn test_double_well_away_from_min() {
        // REQ-AUTO-001: energy > 0 away from minima
        let dw = DoubleWell::new(2);
        let x = array![0.0, 0.0]; // saddle point
        let e = dw.energy(&x.view());
        assert!(e > 0.0, "Energy should be >0 at saddle, got {e}");
    }

    #[test]
    fn test_rosenbrock_at_minimum() {
        // REQ-AUTO-001: Rosenbrock energy = 0 at [1,1,...,1]
        let rb = Rosenbrock::new(4);
        let info = rb.info();
        let e = rb.energy(&info.global_min_location.view());
        assert!(e.abs() < 1e-6, "Energy at minimum should be ~0, got {e}");
    }

    #[test]
    fn test_rosenbrock_gradient_at_minimum() {
        // REQ-AUTO-001: gradient = 0 at minimum
        let rb = Rosenbrock::new(3);
        let x = Array1::ones(3);
        let grad = rb.grad_energy(&x.view());
        assert!(
            grad.iter().all(|g| g.abs() < 1e-4),
            "Gradient at minimum should be ~0, got {:?}",
            grad
        );
    }

    #[test]
    fn test_rosenbrock_away_from_min() {
        // REQ-AUTO-001: energy > 0 away from minimum
        let rb = Rosenbrock::new(2);
        let x = array![0.0, 0.0];
        let e = rb.energy(&x.view());
        assert!(e > 0.0);
        // At [0,0]: 100*(0-0)^2 + (1-0)^2 = 1.0
        assert!((e - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ackley_at_minimum() {
        // REQ-AUTO-001: Ackley energy ≈ 0 at origin
        let ack = Ackley::new(3);
        let info = ack.info();
        let e = ack.energy(&info.global_min_location.view());
        assert!(e.abs() < 1e-4, "Energy at minimum should be ~0, got {e}");
    }

    #[test]
    fn test_ackley_away_from_min() {
        // REQ-AUTO-001: energy > 0 away from origin
        let ack = Ackley::new(2);
        let x = array![1.0, 1.0];
        let e = ack.energy(&x.view());
        assert!(e > 0.0, "Energy away from origin should be >0, got {e}");
    }

    #[test]
    fn test_rastrigin_at_minimum() {
        // REQ-AUTO-001: Rastrigin energy = 0 at origin
        let ras = Rastrigin::new(3);
        let info = ras.info();
        let e = ras.energy(&info.global_min_location.view());
        assert!(e.abs() < 1e-4, "Energy at minimum should be ~0, got {e}");
    }

    #[test]
    fn test_rastrigin_gradient_at_minimum() {
        // REQ-AUTO-001: gradient = 0 at origin
        let ras = Rastrigin::new(2);
        let x = Array1::zeros(2);
        let grad = ras.grad_energy(&x.view());
        assert!(
            grad.iter().all(|g| g.abs() < 1e-4),
            "Gradient at minimum should be ~0, got {:?}",
            grad
        );
    }

    #[test]
    fn test_rastrigin_local_minima() {
        // REQ-AUTO-001: Rastrigin has local minima at integer points
        let ras = Rastrigin::new(1);
        let e_origin = ras.energy(&array![0.0].view());
        let e_one = ras.energy(&array![1.0].view());
        // Origin is global min (energy=0), x=1 is local min (energy=1)
        assert!(e_origin < e_one);
        assert!((e_one - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_gaussian_mixture_two_modes() {
        // REQ-AUTO-001: GMM has low energy near both modes
        let gmm = GaussianMixture::two_modes(4.0);
        let e_mode1 = gmm.energy(&array![-2.0].view());
        let e_mode2 = gmm.energy(&array![2.0].view());
        let e_between = gmm.energy(&array![0.0].view());
        // Energy should be lower at modes than between them
        assert!(
            e_mode1 < e_between,
            "Mode 1 energy {e_mode1} should be < between {e_between}"
        );
        assert!(
            e_mode2 < e_between,
            "Mode 2 energy {e_mode2} should be < between {e_between}"
        );
        // Symmetric modes should have equal energy
        assert!((e_mode1 - e_mode2).abs() < 1e-4);
    }

    #[test]
    fn test_all_benchmarks_implement_energy_function() {
        // REQ-AUTO-001: all benchmarks work through EnergyFunction trait
        let benchmarks: Vec<Box<dyn EnergyFunction>> = vec![
            Box::new(DoubleWell::new(2)),
            Box::new(Rosenbrock::new(2)),
            Box::new(Ackley::new(2)),
            Box::new(Rastrigin::new(2)),
            Box::new(GaussianMixture::two_modes(4.0)),
        ];

        for (i, b) in benchmarks.iter().enumerate() {
            let x = Array1::zeros(b.input_dim());
            let e = b.energy(&x.view());
            assert!(
                e.is_finite(),
                "Benchmark {i} energy should be finite, got {e}"
            );
            let grad = b.grad_energy(&x.view());
            assert!(
                grad.iter().all(|g| g.is_finite()),
                "Benchmark {i} gradient should be finite"
            );
        }
    }
}
