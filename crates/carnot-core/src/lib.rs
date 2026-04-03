//! # carnot-core: Core Energy Based Model traits, types, and abstractions
//!
//! This is the foundation crate for the Carnot framework. If you're new to
//! Energy Based Models, start here — everything else builds on these concepts.
//!
//! ## What is an Energy Based Model?
//!
//! Think of it like a landscape of hills and valleys. Every possible configuration
//! of your data (an image, a Sudoku grid, a robot's joint angles) maps to a point
//! on this landscape. The height at that point is the "energy."
//!
//! - **Low energy** = the configuration is good, valid, likely
//! - **High energy** = the configuration is bad, invalid, unlikely
//!
//! A solved Sudoku has energy 0 (all constraints satisfied). An invalid Sudoku
//! has high energy (the violated constraints push the energy up).
//!
//! ## Why is this different from an LLM?
//!
//! An LLM generates text token-by-token, committing to each word before seeing
//! what comes next. If it makes a mistake early, it can't go back and fix it.
//!
//! An EBM evaluates the *entire configuration at once*. It doesn't generate
//! sequentially — it looks at everything simultaneously and asks "how good is
//! this?" If something is wrong, gradient descent on the energy can fix *just*
//! the broken part without starting over.
//!
//! ## Key types in this crate
//!
//! - [`EnergyFunction`] — the core trait that all models implement
//! - [`ModelState`] — everything needed to save/load a model
//! - [`verify`] — constraint-based verification (the anti-hallucination primitive)
//! - [`benchmarks`] — standard test problems with known solutions
//!
//! Implements: REQ-CORE-001, REQ-CORE-003, REQ-CORE-004, REQ-CORE-006,
//!             REQ-VERIFY-001 through REQ-VERIFY-007

use ndarray::{Array1, ArrayView1, ArrayView2};
use std::collections::HashMap;

pub mod benchmarks;
pub mod error;
pub mod init;
pub mod serialize;
pub mod verify;

pub use error::CarnotError;

// ---------------------------------------------------------------------------
// Numeric precision
// ---------------------------------------------------------------------------

/// The floating-point type used throughout Carnot.
///
/// By default this is `f32` (single precision), which is the standard for
/// machine learning — it's fast, uses half the memory of f64, and is what
/// GPUs are optimized for.
///
/// If you need higher precision (e.g., for numerical stability in very deep
/// networks or for scientific computing where you need ~15 decimal digits
/// instead of ~7), enable the `f64` feature flag:
///
/// ```toml
/// carnot-core = { version = "0.1", features = ["f64"] }
/// ```
///
/// Spec: REQ-CORE-006
#[cfg(not(feature = "f64"))]
pub type Float = f32;
#[cfg(feature = "f64")]
pub type Float = f64;

// ---------------------------------------------------------------------------
// The EnergyFunction trait — the heart of everything
// ---------------------------------------------------------------------------

/// The core trait for energy-based models. Every model in Carnot implements this.
///
/// # What does an energy function do?
///
/// It takes an input vector `x` (which could represent anything — pixel values,
/// Sudoku cell values, robot joint angles) and returns a single number: the energy.
///
/// ```text
/// x = [0.5, 0.3, 0.8]  →  energy(x) = 2.7
/// x = [1.0, 0.0, 0.0]  →  energy(x) = 0.1  ← this configuration is "better"
/// ```
///
/// The energy has no fixed scale or meaning — what matters is the *relative*
/// ordering. Lower energy = better configuration.
///
/// # Why three methods?
///
/// - **`energy(x)`** — "how good is this single configuration?" Used for
///   verification (is the energy low enough?) and for MCMC acceptance decisions.
///
/// - **`energy_batch(xs)`** — same thing but for many configurations at once.
///   This is just a convenience that loops over `energy()` by default, but
///   models can override it for vectorized speedups.
///
/// - **`grad_energy(x)`** — "which direction should I move x to reduce the energy?"
///   This is the gradient (derivative) of the energy with respect to the input.
///   It's the key primitive for both:
///   - **Sampling**: Langevin dynamics uses the gradient to generate samples
///     from the learned distribution
///   - **Repair**: gradient descent on violated constraints to fix a broken
///     configuration (the surgical correction that LLMs can't do)
///
/// # For engineers coming from traditional ML
///
/// If you've used loss functions in neural network training, energy is similar —
/// but with a crucial difference. A loss function measures how wrong your
/// *predictions* are against *labels*. An energy function measures how
/// "valid" a *configuration* is, with no labels needed. The energy landscape
/// itself encodes what's valid and what isn't.
///
/// # Example: implementing a simple energy function
///
/// ```rust,ignore
/// struct QuadraticEnergy;
///
/// impl EnergyFunction for QuadraticEnergy {
///     fn energy(&self, x: &ArrayView1<Float>) -> Float {
///         // E(x) = 0.5 * ||x||^2
///         // This defines a bowl-shaped landscape with minimum at the origin.
///         // Low energy near [0,0,...], high energy far from origin.
///         0.5 * x.dot(x)
///     }
///
///     fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
///         // The gradient of 0.5*||x||^2 is just x itself.
///         // It points away from the minimum — so to minimize,
///         // you subtract the gradient: x_new = x - step_size * grad.
///         x.to_owned()
///     }
///
///     fn input_dim(&self) -> usize { 0 } // accepts any dimension
/// }
/// ```
///
/// Spec: REQ-CORE-001
pub trait EnergyFunction: Send + Sync {
    /// Compute the scalar energy for a single input vector.
    ///
    /// This is the fundamental operation — it answers "how good is this
    /// configuration?" The returned value should be finite (not NaN or infinity)
    /// for any valid input.
    ///
    /// # Arguments
    /// * `x` - A 1-D array view representing one configuration.
    ///   For example, a flattened 28x28 MNIST image would be a 784-element vector.
    ///
    /// # Returns
    /// A scalar energy value. Lower = better/more likely configuration.
    fn energy(&self, x: &ArrayView1<Float>) -> Float;

    /// Compute energy for a batch of inputs at once.
    ///
    /// The default implementation just loops over each row and calls `energy()`.
    /// Models can override this for vectorized implementations (e.g., using SIMD
    /// or matrix operations to process the whole batch in one shot).
    ///
    /// # Arguments
    /// * `xs` - A 2-D array where each row is one input vector.
    ///   Shape: (batch_size, input_dim).
    ///
    /// # Returns
    /// A 1-D array of energies, one per input. Length = batch_size.
    fn energy_batch(&self, xs: &ArrayView2<Float>) -> Array1<Float> {
        Array1::from_iter(xs.rows().into_iter().map(|row| self.energy(&row)))
    }

    /// Compute the gradient of the energy with respect to the input.
    ///
    /// The gradient tells you "which direction increases the energy the fastest."
    /// To *decrease* the energy (find better configurations), you move in the
    /// *opposite* direction of the gradient:
    ///
    /// ```text
    /// x_improved = x - step_size * grad_energy(x)
    /// ```
    ///
    /// This is used everywhere in Carnot:
    /// - **Langevin dynamics**: adds noise to gradient descent to sample from
    ///   the energy distribution (not just find the minimum)
    /// - **HMC**: uses the gradient as a "force" in a physics simulation
    /// - **Repair**: descends on violated constraints to fix broken configurations
    /// - **Training**: score matching uses the gradient as the learning target
    ///
    /// # Arguments
    /// * `x` - A 1-D array view representing one configuration.
    ///
    /// # Returns
    /// A 1-D array of the same shape as `x`, where each element is
    /// ∂E/∂x_i (the partial derivative of energy with respect to that input dimension).
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float>;

    /// The number of input dimensions this model expects.
    ///
    /// For example, an Ising model built for MNIST would return 784 (28*28 pixels).
    /// Return 0 if the model accepts any dimension.
    fn input_dim(&self) -> usize;
}

// ---------------------------------------------------------------------------
// Model configuration and state
// ---------------------------------------------------------------------------

/// Configuration that describes a model's architecture.
///
/// This is the "blueprint" — it tells you the shape of the model (how many
/// input dimensions, how many hidden layers, what precision) but doesn't
/// contain the actual learned parameter values.
///
/// Think of it like a recipe vs. the cooked dish: ModelConfig is the recipe,
/// ModelState (below) is the dish with all the specific ingredient amounts.
///
/// Spec: REQ-CORE-003
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ModelConfig {
    /// Number of input dimensions (e.g., 784 for MNIST, 81 for Sudoku).
    pub input_dim: usize,
    /// Sizes of hidden layers. For example, [512, 256] means two hidden layers
    /// with 512 and 256 neurons respectively. Empty for models without hidden
    /// layers (like the Ising model's direct pairwise interactions).
    pub hidden_dims: Vec<usize>,
    /// Whether to use 32-bit or 64-bit floating point numbers.
    pub precision: Precision,
}

/// Whether to use single-precision (f32) or double-precision (f64) math.
///
/// f32 is the default and the standard for machine learning:
/// - Uses half the memory of f64
/// - Roughly 2x faster on most hardware
/// - ~7 decimal digits of precision (plenty for ML)
///
/// f64 is useful when you need:
/// - ~15 decimal digits of precision
/// - Better numerical stability for very deep networks
/// - Scientific computing accuracy
///
/// Spec: REQ-CORE-006
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Precision {
    #[default]
    F32,
    F64,
}

/// Metadata about a model's training history.
///
/// This tracks how far along training is and how the loss has been changing.
/// It's stored alongside the model parameters so you can resume training
/// from a checkpoint without losing track of where you left off.
///
/// Spec: REQ-CORE-003
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ModelMetadata {
    /// How many training steps have been completed.
    /// A "step" is one parameter update (one batch of data processed).
    pub step: u64,
    /// The loss value at each training step. Useful for plotting training curves
    /// to see if the model is still improving or has converged.
    pub loss_history: Vec<Float>,
}

/// Everything needed to save, load, and resume a model.
///
/// This bundles together:
/// - The actual learned parameter values (weights and biases)
/// - The architecture configuration (so you know how to reconstruct the model)
/// - Training metadata (so you can resume training from where you left off)
///
/// Saved to disk using the safetensors format (not pickle!) for cross-language
/// compatibility and security. A model saved from Rust can be loaded in Python
/// and vice versa.
///
/// Spec: REQ-CORE-003
#[derive(Debug, Clone)]
pub struct ModelState {
    /// The learned parameters, keyed by name.
    /// For example: {"weight_0": [0.1, -0.3, ...], "bias_0": [0.0, 0.1, ...]}
    pub parameters: HashMap<String, Array1<Float>>,
    /// The architecture blueprint — needed to reconstruct the model structure.
    pub config: ModelConfig,
    /// Training progress — step count and loss history.
    pub metadata: ModelMetadata,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A trivial energy function for testing: E(x) = 0.5 * ||x||^2
    ///
    /// This creates a simple bowl-shaped energy landscape where the minimum
    /// is at the origin [0, 0, ..., 0]. It's the energy function equivalent
    /// of a standard Gaussian distribution — samples from this energy landscape
    /// would cluster around zero with variance 1.
    struct QuadraticEnergy;

    impl EnergyFunction for QuadraticEnergy {
        // SCENARIO-CORE-001: Compute Energy for Single Input
        fn energy(&self, x: &ArrayView1<Float>) -> Float {
            // Dot product of x with itself gives ||x||^2 (sum of squares).
            // Multiply by 0.5 so the gradient is simply x (not 2x).
            0.5 * x.dot(x)
        }

        // SCENARIO-CORE-003: Compute Energy Gradient
        fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
            // For E(x) = 0.5 * ||x||^2, the gradient dE/dx = x.
            // This points away from the origin — gradient descent would
            // move toward the origin (the minimum).
            x.to_owned()
        }

        fn input_dim(&self) -> usize {
            0 // accepts any dimension
        }
    }

    #[test]
    fn test_quadratic_energy_single() {
        // SCENARIO-CORE-001
        // E([1, 2, 3]) = 0.5 * (1 + 4 + 9) = 7.0
        let model = QuadraticEnergy;
        let x = array![1.0, 2.0, 3.0];
        let e = model.energy(&x.view());
        assert!((e - 7.0).abs() < 1e-6);
        assert!(e.is_finite());
    }

    #[test]
    fn test_quadratic_energy_batch() {
        // SCENARIO-CORE-002
        // Two inputs: [1,0] has energy 0.5, [0,2] has energy 2.0
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
        // Verify our analytical gradient matches a numerical approximation.
        // This "finite difference" check is the standard way to verify gradients:
        // compute (E(x+eps) - E(x-eps)) / (2*eps) and compare to the analytical value.
        let model = QuadraticEnergy;
        let x = array![1.0, 2.0, 3.0];
        let grad = model.grad_energy(&x.view());
        assert_eq!(grad.len(), x.len());

        let eps: Float = 1e-3;
        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[i] += eps;
            x_minus[i] -= eps;
            let e_plus = model.energy(&x_plus.view());
            let e_minus = model.energy(&x_minus.view());
            // This is the "central finite difference" approximation of the derivative
            let fd = (e_plus - e_minus) / (2.0 * eps);
            assert!(
                (grad[i] - fd).abs() < 0.1,
                "Gradient mismatch at index {i}: analytic={}, fd={fd}",
                grad[i]
            );
        }
    }

    #[test]
    fn test_precision_default() {
        // REQ-CORE-006
        assert_eq!(Precision::default(), Precision::F32);
    }
}
