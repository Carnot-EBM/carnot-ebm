//! # Analytical Benchmark Energy Functions with Known Solutions
//!
//! **For researchers:** A suite of classical optimization test functions (DoubleWell, Rosenbrock,
//! Ackley, Rastrigin, Gaussian mixture) implementing [`EnergyFunction`], with known global minima
//! for quantitative evaluation of samplers and optimizers. Feeds into the autoresearch loop
//! (REQ-AUTO-001).
//!
//! **For engineers coming from neural networks:**
//!
//! When developing MCMC samplers or training algorithms for Energy-Based Models, you need a way
//! to know if your code is *actually working*. With real-world energy functions (like learned
//! neural network potentials), you usually don't know the correct answer ahead of time. That
//! makes debugging nearly impossible — is your sampler broken, or is the energy landscape
//! just hard?
//!
//! **Benchmark energy functions solve this problem.** They are simple mathematical functions
//! where we know *exactly* where the minimum is, what the minimum energy value is, and what
//! the distribution should look like. If your sampler can't find the known minimum of a
//! 2D Rosenbrock function, it definitely won't work on a 1000-dimensional Boltzmann machine.
//!
//! Each benchmark tests a different challenge that real energy landscapes present:
//!
//! | Benchmark | What it tests | Real-world analogy |
//! |-----------|--------------|-------------------|
//! | [`DoubleWell`] | **Multimodality** — can the sampler find *both* low-energy regions? | An EBM trained on cats AND dogs has two modes |
//! | [`Rosenbrock`] | **Narrow valleys** — can the sampler navigate curved, thin regions? | Highly correlated parameters in a model |
//! | [`Ackley`] | **Many local minima** — can the sampler escape traps? | Noisy energy landscapes with many spurious modes |
//! | [`Rastrigin`] | **Regular local minima grid** — systematic trap avoidance | Periodic structures in the energy surface |
//! | [`GaussianMixture`] | **Known distribution** — does the sampler recover correct probabilities? | The gold standard: we know the exact target distribution |
//!
//! All benchmarks implement the [`EnergyFunction`] trait, so they plug directly into any
//! sampler via the [`Sampler`](crate::Sampler) interface without modification.
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
///
/// **For researchers:** Stores the ground-truth minimum for quantitative evaluation:
/// energy gap = `sampler_energy - global_min_energy`.
///
/// **For engineers:** Every benchmark comes with a "cheat sheet" — this struct tells you
/// exactly where the minimum is and what energy value it has. When you run a sampler or
/// optimizer, you compare its result against this known answer to measure how well it did.
///
/// For example, the Rosenbrock function has `global_min_energy = 0.0` at
/// `global_min_location = [1, 1, ..., 1]`. If your optimizer returns energy 0.003
/// at position [0.98, 0.97, ...], you know it got close but not perfect.
#[derive(Debug, Clone)]
pub struct BenchmarkInfo {
    /// Human-readable name for this benchmark (e.g., "double_well", "rosenbrock").
    pub name: &'static str,
    /// Dimensionality of the input space.
    pub input_dim: usize,
    /// The energy value at the global minimum (usually 0.0 for these benchmarks).
    pub global_min_energy: Float,
    /// The exact location of the global minimum in input space.
    pub global_min_location: Array1<Float>,
    /// A short text description of the energy function formula and properties.
    pub description: &'static str,
}

/// Double-well potential: tests multimodal sampling.
///
/// **For researchers:** Quartic double-well in the first coordinate with quadratic
/// confinement in remaining dimensions. `E(x) = (x[0]^2 - 1)^2 + sum_{i>0} x[i]^2`.
/// Two symmetric global minima at `x = [+/-1, 0, ..., 0]` with `E = 0`. The barrier
/// at x[0]=0 has height 1.0. Tests inter-modal mixing of MCMC samplers.
///
/// **For engineers coming from neural networks:**
///
/// Imagine a landscape with two valleys (like a "W" shape when viewed from the side).
/// The two valleys are at x[0] = -1 and x[0] = +1, both with energy = 0. Between them
/// is a hill at x[0] = 0 with energy = 1.
///
/// This tests the hardest problem in MCMC sampling: **can the sampler jump between the
/// two valleys?** A naive sampler might get stuck in one valley and never discover the
/// other. Good samplers (especially HMC with enough momentum) can cross the barrier.
///
/// For example: if you sample 10,000 points and they're ALL near x[0]=+1, the sampler
/// failed to discover the equally valid x[0]=-1 valley. A working sampler should visit both.
///
/// In the real world, this models situations like: an image generation model that should
/// produce BOTH cats and dogs, not just one category.
///
/// Spec: REQ-AUTO-001
pub struct DoubleWell {
    /// Dimensionality of the input space. Must be >= 1.
    /// Only the first dimension has the double-well structure; remaining dimensions
    /// are simple quadratic (Gaussian) confinement.
    dim: usize,
}

impl DoubleWell {
    /// Create a new DoubleWell benchmark with the given dimensionality.
    ///
    /// # Panics
    /// Panics if `dim < 1` — the double-well requires at least one dimension for
    /// the quartic potential.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "DoubleWell requires dim >= 1");
        Self { dim }
    }

    /// Return metadata including the known global minimum.
    ///
    /// Note: only one of the two symmetric minima is returned (the one at x[0]=+1).
    /// The other minimum at x[0]=-1 has the same energy.
    pub fn info(&self) -> BenchmarkInfo {
        let mut loc = Array1::zeros(self.dim);
        loc[0] = 1.0; // One of the two symmetric minima
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
        // The "double well" part: (x[0]^2 - 1)^2 is zero when x[0] = +/-1
        // and equals 1 when x[0] = 0 (the barrier between wells).
        let well = (x[0] * x[0] - 1.0) * (x[0] * x[0] - 1.0);
        // The "confinement" part: simple quadratic in all other dimensions,
        // pulling them toward zero. This keeps the problem well-behaved.
        let rest: Float = x.iter().skip(1).map(|&xi| xi * xi).sum();
        well + rest
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut grad = Array1::zeros(x.len());
        // Gradient of (x0^2 - 1)^2 using chain rule:
        // d/dx0 [(x0^2 - 1)^2] = 2*(x0^2 - 1) * 2*x0 = 4*x0*(x0^2 - 1)
        grad[0] = 4.0 * x[0] * (x[0] * x[0] - 1.0);
        // Gradient of xi^2 is simply 2*xi for each remaining dimension
        for i in 1..x.len() {
            grad[i] = 2.0 * x[i];
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Rosenbrock function: tests navigation of narrow curved valleys.
///
/// **For researchers:** The classic Rosenbrock "banana" function.
/// `E(x) = sum_{i=0}^{n-2} [100*(x[i+1] - x[i]^2)^2 + (1 - x[i])^2]`.
/// Global minimum at `x = [1, 1, ..., 1]` with `E = 0`. The narrow, curved valley
/// makes this a challenging optimization target with high condition number.
///
/// **For engineers coming from neural networks:**
///
/// The Rosenbrock function is one of the most famous test problems in optimization. Its
/// energy landscape looks like a long, narrow, curved valley (shaped like a banana when
/// viewed from above in 2D). Finding that the minimum is at [1, 1, ..., 1] is easy if
/// you just read the formula. But for an algorithm that can only take local steps, the
/// narrow valley is treacherous:
///
/// - **The valley is easy to find** — most starting points quickly descend into it.
/// - **The minimum within the valley is hard to reach** — the valley curves, and the
///   floor slopes very gently, so progress along it is extremely slow.
///
/// This tests whether your sampler or optimizer can handle **highly correlated dimensions**
/// (x[i+1] wants to be close to x[i]^2, creating tight coupling between adjacent variables).
///
/// In the real world, this models situations like: model parameters that are tightly coupled
/// (e.g., a weight and its corresponding bias), where changing one without the other leads
/// to bad results, but the "good" direction is hard to find.
///
/// Spec: REQ-AUTO-001
pub struct Rosenbrock {
    /// Dimensionality of the input space. Must be >= 2.
    /// The function sums over consecutive pairs of dimensions,
    /// so at least two dimensions are required.
    dim: usize,
}

impl Rosenbrock {
    /// Create a new Rosenbrock benchmark with the given dimensionality.
    ///
    /// # Panics
    /// Panics if `dim < 2` — the Rosenbrock function is defined over pairs of coordinates.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 2, "Rosenbrock requires dim >= 2");
        Self { dim }
    }

    /// Return metadata including the known global minimum at [1, 1, ..., 1].
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
            // The "banana" term: 100 * (x[i+1] - x[i]^2)^2
            // This creates a curved valley where x[i+1] wants to equal x[i]^2.
            // The factor of 100 makes the valley walls very steep (high curvature
            // perpendicular to the valley) while the valley floor is nearly flat.
            let a = x[i + 1] - x[i] * x[i];
            // The "drift" term: (1 - x[i])^2
            // This gently pulls each x[i] toward 1, creating the minimum at [1,1,...,1].
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
            // Gradient w.r.t. x[i] from the i-th pair:
            // d/dx[i] [100*(x[i+1] - x[i]^2)^2] = -400*x[i]*(x[i+1] - x[i]^2)
            // d/dx[i] [(1 - x[i])^2] = 2*(x[i] - 1)
            grad[i] += -400.0 * x[i] * a + 2.0 * (x[i] - 1.0);
            // Gradient w.r.t. x[i+1] from the i-th pair:
            // d/dx[i+1] [100*(x[i+1] - x[i]^2)^2] = 200*(x[i+1] - x[i]^2)
            grad[i + 1] += 200.0 * a;
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Ackley function: tests escape from many local minima.
///
/// **For researchers:** Multimodal function with a nearly flat outer region and a large hole
/// at the origin. `E(x) = -20*exp(-0.2*sqrt(mean(x^2))) - exp(mean(cos(2*pi*x))) + 20 + e`.
/// Global minimum at origin with `E = 0`. The cosine term creates a dense grid of local minima
/// that traps gradient-based methods. Uses numerical gradient (central differences).
///
/// **For engineers coming from neural networks:**
///
/// The Ackley function looks like a bumpy, crinkled surface with one deep hole at the origin.
/// Away from the origin, the surface is nearly flat but covered with small bumps (local minima)
/// created by the cosine terms. Think of it like a golf course with many sand traps — the
/// global minimum (the hole) is at the center, but a ball rolling on the surface keeps getting
/// caught in the traps.
///
/// This tests whether your sampler can **escape local minima**. A simple gradient descent
/// would get stuck in the nearest bump. MCMC samplers use randomness to hop between bumps,
/// and the best ones (like well-tuned HMC) can reach the global minimum reliably.
///
/// The gradient for Ackley is computed **numerically** (via central finite differences)
/// rather than analytically, because the analytical gradient involves a messy combination
/// of exponentials and trigonometric functions. This is slower but simpler and less error-prone.
///
/// Spec: REQ-AUTO-001
pub struct Ackley {
    /// Dimensionality of the input space. Must be >= 1.
    dim: usize,
}

impl Ackley {
    /// Create a new Ackley benchmark with the given dimensionality.
    ///
    /// # Panics
    /// Panics if `dim < 1`.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "Ackley requires dim >= 1");
        Self { dim }
    }

    /// Return metadata including the known global minimum at the origin.
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
        // First term: based on the RMS (root mean square) distance from the origin.
        // The exp(-0.2 * sqrt(mean(x^2))) part creates a broad bowl centered at the origin.
        let sum_sq: Float = x.iter().map(|&xi| xi * xi).sum();
        // Second term: based on the mean of cosines. The cos(2*pi*xi) terms create the
        // regular grid of bumps (local minima). At the origin, all cosines = 1 (maximum),
        // which makes this term most negative, creating the global minimum.
        let sum_cos: Float = x.iter().map(|&xi| (2.0 * PI * xi).cos()).sum();
        // Combine: the constants 20 and e are chosen so that E(0) = 0.
        -20.0 * (-0.2 * (sum_sq / n).sqrt()).exp() - (sum_cos / n).exp()
            + 20.0
            + std::f32::consts::E
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Numerical gradient via central finite differences: grad[i] ≈ (E(x+eps_i) - E(x-eps_i)) / (2*eps)
        // This is O(eps^2) accurate. We use eps = 1e-4 as a balance between truncation
        // error (smaller eps = more accurate formula) and floating-point roundoff (smaller
        // eps = more cancellation error, especially with f32).
        let eps: Float = 1e-4;
        let mut grad = Array1::zeros(x.len());
        for i in 0..x.len() {
            let mut x_p = x.to_owned();
            let mut x_m = x.to_owned();
            x_p[i] += eps; // Perturb dimension i forward
            x_m[i] -= eps; // Perturb dimension i backward
            grad[i] = (self.energy(&x_p.view()) - self.energy(&x_m.view())) / (2.0 * eps);
        }
        grad
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Rastrigin function: tests systematic local minima avoidance.
///
/// **For researchers:** Highly multimodal with `10^n` local minima on a regular grid.
/// `E(x) = 10*n + sum(x[i]^2 - 10*cos(2*pi*x[i]))`. Global minimum at origin with `E = 0`.
/// Local minima at every integer lattice point. The cosine modulation has amplitude 10,
/// making basins deep and hard to escape via gradient-only methods.
///
/// **For engineers coming from neural networks:**
///
/// The Rastrigin function is like a waffle iron — a regular grid of equally-spaced bumps
/// covering the entire space, with the deepest point at the origin. Unlike Ackley (which
/// has one deep hole surrounded by shallow bumps), Rastrigin has deep local minima at EVERY
/// integer point: (0,0), (1,0), (0,1), (1,1), (-1,0), etc.
///
/// In 2D there are already hundreds of local minima. In 10D there are 10 billion. This
/// is the ultimate test of whether a sampler can systematically explore the space rather
/// than getting trapped. Simple gradient descent will always get stuck at the nearest
/// integer point.
///
/// The energy at the origin is 0, at x=[1,0,...] it's 1.0, at x=[2,0,...] it's 4.0, etc.
/// The cosine terms create the local minima; the quadratic x^2 term creates the overall
/// bowl shape that makes the origin the global minimum.
///
/// Spec: REQ-AUTO-001
pub struct Rastrigin {
    /// Dimensionality of the input space. Must be >= 1.
    dim: usize,
}

impl Rastrigin {
    /// Create a new Rastrigin benchmark with the given dimensionality.
    ///
    /// # Panics
    /// Panics if `dim < 1`.
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 1, "Rastrigin requires dim >= 1");
        Self { dim }
    }

    /// Return metadata including the known global minimum at the origin.
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
        // Each dimension contributes xi^2 (bowl shape) minus 10*cos(2*pi*xi) (periodic bumps).
        // The 10*n constant offsets the cosine contributions so that E(0) = 0.
        let sum: Float = x
            .iter()
            .map(|&xi| xi * xi - 10.0 * (2.0 * PI * xi).cos())
            .sum();
        10.0 * n + sum
    }

    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Analytical gradient: d/dxi [xi^2 - 10*cos(2*pi*xi)] = 2*xi + 10*2*pi*sin(2*pi*xi)
        // The sin term creates oscillating gradient that points toward the nearest integer,
        // which is why gradient descent gets trapped at integer points.
        Array1::from_iter(
            x.iter()
                .map(|&xi| 2.0 * xi + 10.0 * 2.0 * PI * (2.0 * PI * xi).sin()),
        )
    }

    fn input_dim(&self) -> usize {
        self.dim
    }
}

/// Gaussian mixture energy: the gold-standard benchmark with a known target distribution.
///
/// **For researchers:** Negative log-likelihood of a Gaussian mixture model:
/// `E(x) = -log(sum_k w_k * N(x; mu_k, sigma_k^2 * I))`. Implements log-sum-exp for
/// numerical stability. Known parameters allow exact computation of KL divergence,
/// mode coverage, and other distributional metrics. Uses numerical gradient.
///
/// **For engineers coming from neural networks:**
///
/// This is the most important benchmark for MCMC samplers because we know the *exact*
/// target distribution, not just the minimum. A Gaussian mixture is a sum of several
/// bell curves (Gaussians), each centered at a different location ("mode"). The energy
/// is the negative log-probability: low energy = high probability.
///
/// For example, `GaussianMixture::two_modes(4.0)` creates two equal-weight Gaussians
/// centered at x = -2 and x = +2. A correct sampler should produce:
/// - ~50% of samples near x = -2
/// - ~50% of samples near x = +2
/// - Very few samples near x = 0 (between the modes, low probability)
///
/// If your sampler produces 90% of samples near one mode and 10% near the other, it has
/// a **mode imbalance** problem — it's not mixing well between the two regions.
///
/// Unlike the other benchmarks which only test "can you find the minimum?", this one
/// tests "can you sample from the correct *distribution*?" — which is the actual goal
/// of MCMC in EBMs.
///
/// Spec: REQ-AUTO-001
pub struct GaussianMixture {
    /// Dimensionality of the input space.
    dim: usize,
    /// Center (mean) of each Gaussian component.
    means: Vec<Array1<Float>>,
    /// Variance (sigma^2) of each component. Each component is isotropic (same variance
    /// in all dimensions), so this is a scalar per component rather than a full covariance matrix.
    variances: Vec<Float>,
    /// Mixing weight of each component. Must sum to 1.0. Determines what fraction of
    /// samples should come from each Gaussian. Equal weights (e.g., [0.5, 0.5]) mean
    /// each mode is equally important.
    weights: Vec<Float>,
}

impl GaussianMixture {
    /// Create a simple 1D mixture of two Gaussians separated by `separation` units.
    ///
    /// **For engineers:** This is the simplest multimodal test case. The two modes are
    /// placed symmetrically around the origin: one at `-separation/2`, one at `+separation/2`.
    /// Both have unit variance and equal weight. Larger separation makes mixing harder
    /// because the sampler has to traverse a wider low-probability region between modes.
    ///
    /// For example: `two_modes(4.0)` puts modes at -2.0 and +2.0, which is 4 standard
    /// deviations apart — quite challenging for most samplers.
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

    /// Return metadata about this mixture model.
    ///
    /// Note: `global_min_energy` is approximate (0.0), and `global_min_location` returns
    /// just the first mode's center. The true distribution has multiple modes — use
    /// distributional metrics (KL divergence, mode coverage) for proper evaluation.
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
        // We compute E(x) = -log p(x) where p(x) = sum_k w_k * N(x; mu_k, sigma_k^2).
        // Direct computation of the sum would cause numerical underflow (tiny probabilities),
        // so we use the log-sum-exp trick: log(sum(exp(a_k))) = max(a_k) + log(sum(exp(a_k - max(a_k)))).

        // Accumulate log(sum of weighted Gaussian densities) using log-sum-exp.
        let mut log_sum: Float = Float::NEG_INFINITY;
        for (mean, (&var, &w)) in self
            .means
            .iter()
            .zip(self.variances.iter().zip(self.weights.iter()))
        {
            // Compute the log of the k-th weighted Gaussian component:
            // log(w_k * N(x; mu_k, var_k)) = log(w_k) + log(N(x; mu_k, var_k))
            let diff = x.to_owned() - mean;
            let exponent = -0.5 * diff.dot(&diff) / var; // Mahalanobis distance (isotropic)
            let log_component =
                w.ln() + exponent - 0.5 * (self.dim as Float) * (2.0 * PI * var).ln();

            // Incremental log-sum-exp: combine this component with the running total.
            // This is numerically stable because we always work in log space.
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
        // Numerical gradient via central finite differences.
        // An analytical gradient is possible but involves weighted sums of (x - mu_k) terms
        // with softmax-like weights, which is error-prone to implement. Since this is a
        // benchmark (not inner-loop production code), the numerical approach is preferred.
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
