//! # carnot-ising: Ising (Small Tier) Energy-Based Model
//!
//! ## For Researchers
//!
//! Implements a pairwise interaction energy model (Ising-type Hamiltonian):
//!
//!   E(x) = -0.5 * x^T J x - b^T x
//!
//! where J is a symmetric coupling matrix and b is an external bias field.
//! This is the lightest model tier in Carnot, suitable for edge deployment
//! and pedagogical use. Gradient is computed analytically as dE/dx = -Jx - b.
//!
//! ## For Engineers Learning EBMs
//!
//! An Energy-Based Model (EBM) assigns a single scalar number -- the "energy" --
//! to every possible input. Low energy = the model thinks this input is likely/good.
//! High energy = the model thinks this input is unlikely/bad.
//!
//! The Ising model is the *simplest* EBM in Carnot. It is inspired by the Ising model
//! from statistical physics, where particles (like tiny magnets) interact with their
//! neighbors. Here, instead of a physical lattice, we have a vector of variables and
//! a matrix describing how each pair interacts.
//!
//! ### The energy formula, broken down:
//!
//! ```text
//! E(x) = -0.5 * x^T J x  -  b^T x
//!         ^^^^^^^^^^^^^^^^    ^^^^^^
//!         pairwise term       bias term
//! ```
//!
//! **Pairwise term** (`-0.5 * x^T J x`): Think of magnets on a grid. Each magnet
//! influences its neighbors. The coupling matrix `J` encodes the strength of the
//! interaction between every pair of variables. If `J[i][j]` is large and positive,
//! variables `i` and `j` "want" to have the same sign. The `-0.5` is a convention
//! that makes the gradient clean and accounts for double-counting (since J[i][j]
//! and J[j][i] both contribute).
//!
//! **Bias term** (`-b^T x`): This is like an external magnetic field pushing each
//! variable toward a preferred value. If `b[i]` is positive, the energy is lower
//! (more favorable) when `x[i]` is positive.
//!
//! ### Why is the coupling matrix symmetric?
//!
//! The interaction between variable i and variable j must be the same as the
//! interaction between j and i (Newton's third law, if you like). Mathematically,
//! J[i][j] == J[j][i]. We enforce this by averaging: J = (J + J^T) / 2.
//!
//! ### When to use this model
//!
//! - Learning how EBMs work (this is the "Hello World" of energy models)
//! - Lightweight edge deployment where model size matters (< 10MB for 784 dims)
//! - Problems with known pairwise structure (e.g., Markov Random Fields)
//! - As a fast baseline before scaling up to Gibbs or Boltzmann tiers
//!
//! For engineers coming from neural networks: this model has NO hidden layers
//! and NO nonlinear activations. The energy is a simple quadratic function of
//! the input. Think of it as a single "layer" that outputs a scalar.
//!
//! Spec: REQ-TIER-001, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006

use carnot_core::init::Initializer;
use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for the Ising model.
///
/// # For Researchers
///
/// Parameterizes the Ising Hamiltonian: input dimensionality, optional
/// hidden projection, and coupling matrix initialization strategy.
///
/// # For Engineers
///
/// This struct holds all the knobs you can turn when creating an Ising model:
///
/// - `input_dim`: How many variables does your input vector have? For example,
///   a 28x28 grayscale image (like MNIST) has 784 pixels, so `input_dim = 784`.
///
/// - `hidden_dim`: Optional. If set, the model would project to a hidden space
///   first. Currently, direct pairwise interactions only (no hidden projection
///   implemented in the energy computation). Reserved for future extensions.
///
/// - `coupling_init`: How to initialize the coupling matrix J. Options:
///   - `"xavier_uniform"`: Good default; scales values based on matrix size so
///     energies start in a reasonable range.
///   - `"he_normal"`: Similar but uses a normal distribution; common in deep learning.
///   - `"zeros"`: All couplings start at zero (energy = 0 for all inputs initially).
///
/// For example:
/// ```rust
/// use carnot_ising::IsingConfig;
/// let config = IsingConfig {
///     input_dim: 100,
///     hidden_dim: None,
///     coupling_init: "xavier_uniform".to_string(),
/// };
/// ```
///
/// Spec: REQ-TIER-005
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IsingConfig {
    /// The number of dimensions in the input vector.
    ///
    /// For example, a flattened 28x28 image has input_dim = 784.
    /// Must be greater than zero.
    pub input_dim: usize,
    /// Optional hidden dimension for a projected coupling space.
    /// If None, direct pairwise interactions only (the standard Ising formulation).
    /// If Some(h), reserved for future hidden-projection extensions.
    pub hidden_dim: Option<usize>,
    /// Initialization strategy name for the coupling matrix J.
    ///
    /// Supported values: "xavier_uniform", "he_normal", "zeros".
    /// Xavier uniform is recommended for most use cases -- it scales the
    /// initial coupling strengths so that energies are neither too large
    /// nor too small at the start of training.
    pub coupling_init: String,
}

impl IsingConfig {
    /// Validate that this configuration is well-formed.
    ///
    /// # For Researchers
    ///
    /// Guards against degenerate configurations (zero-dimensional spaces).
    ///
    /// # For Engineers
    ///
    /// This checks that:
    /// - `input_dim > 0` (you need at least one variable)
    /// - If `hidden_dim` is specified, it must also be > 0
    ///
    /// Returns `Err(CarnotError::InvalidConfig(...))` with a human-readable
    /// message if validation fails.
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
    /// Creates a default Ising configuration.
    ///
    /// Defaults: input_dim=784 (MNIST-sized), no hidden dim, xavier_uniform init.
    ///
    /// For engineers: these defaults are chosen so you can create a model and
    /// immediately test it with MNIST-like data without any configuration.
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dim: None,
            coupling_init: "xavier_uniform".to_string(),
        }
    }
}

/// The Ising Energy-Based Model.
///
/// # For Researchers
///
/// Stores a symmetric coupling matrix J and bias vector b.
/// Energy is computed as E(x) = -0.5 * x^T J x - b^T x.
/// Gradient is dE/dx = -Jx - b (exact, no approximation).
/// Parameter count: dim^2 (coupling) + dim (bias).
///
/// # For Engineers
///
/// This is the actual model object. After you create it from an `IsingConfig`,
/// it holds two things:
///
/// 1. **The coupling matrix `J`** (a 2D array, shape [dim, dim]): This is the
///    "brain" of the model. Each entry `J[i][j]` says how strongly variable `i`
///    and variable `j` interact. Think of it like a social network where each
///    person (variable) has a relationship strength with every other person.
///
/// 2. **The bias vector `b`** (a 1D array, length dim): This nudges each
///    variable toward positive or negative values independently of the others.
///
/// For example, with a 3-dimensional input:
/// - `coupling` is a 3x3 matrix (9 parameters)
/// - `bias` is a length-3 vector (3 parameters)
/// - Total: 12 parameters
///
/// For engineers coming from deep learning: this is like a single linear layer
/// with no activation function, except the output is always a scalar energy
/// value rather than a vector of class probabilities.
///
/// Spec: REQ-TIER-001
pub struct IsingModel {
    /// The coupling matrix J (always symmetric, shape [input_dim, input_dim]).
    ///
    /// J[i][j] represents the interaction strength between variables i and j.
    /// Positive values mean the variables "want" to have the same sign.
    /// Negative values mean they "want" to have opposite signs.
    pub coupling: Array2<Float>,
    /// The bias vector b (length input_dim).
    ///
    /// Each entry b[i] shifts the preferred value of variable x[i].
    /// Initialized to zeros (no initial preference).
    pub bias: Array1<Float>,
    /// The configuration that was used to create this model.
    pub config: IsingConfig,
}

impl IsingModel {
    /// Create a new Ising model with the given configuration.
    ///
    /// # For Researchers
    ///
    /// Initializes J using the specified strategy, then symmetrizes via
    /// J = (J + J^T) / 2. Bias is initialized to zero.
    ///
    /// # For Engineers
    ///
    /// This constructor does three things:
    ///
    /// 1. **Validates** the config (returns an error if input_dim is 0, etc.)
    ///
    /// 2. **Creates the coupling matrix** using the chosen initialization strategy,
    ///    then forces it to be symmetric. Why symmetric? Because the energy formula
    ///    x^T J x counts each pair (i,j) twice (once as i*J[i][j]*j and once as
    ///    j*J[j][i]*i), so J[i][j] must equal J[j][i] for the math to be consistent.
    ///    We enforce this by averaging: J_new = (J + J_transposed) / 2.
    ///
    /// 3. **Creates the bias vector** initialized to all zeros (no initial preference).
    ///
    /// For example:
    /// ```rust
    /// use carnot_ising::{IsingConfig, IsingModel};
    /// let model = IsingModel::new(IsingConfig {
    ///     input_dim: 10,
    ///     hidden_dim: None,
    ///     coupling_init: "xavier_uniform".to_string(),
    /// }).unwrap();
    /// assert_eq!(model.coupling.shape(), &[10, 10]);
    /// assert_eq!(model.bias.len(), 10);
    /// ```
    ///
    /// Spec: REQ-TIER-001, REQ-TIER-005, SCENARIO-TIER-006
    pub fn new(config: IsingConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        // Parse the initialization strategy name into an Initializer enum.
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

        // Create the initial coupling matrix (may not be symmetric yet).
        let mut coupling = init.init_matrix(dim, dim);

        // Force symmetry: J = (J + J^T) / 2.
        // This ensures J[i][j] == J[j][i] for all i, j, which is required
        // because the pairwise energy x^T J x is only well-defined (as a
        // quadratic form) when J is symmetric. Without this step, the
        // energy would depend on which "direction" you read the interaction.
        let coupling_t = coupling.t().to_owned();
        coupling = (&coupling + &coupling_t) / 2.0;

        // Bias starts at zero -- no initial preference for any variable.
        let bias = Array1::zeros(dim);

        Ok(Self {
            coupling,
            bias,
            config,
        })
    }

    /// Compute the memory footprint of this model's parameters in bytes.
    ///
    /// # For Researchers
    ///
    /// Returns sizeof(Float) * (dim^2 + dim) -- the coupling matrix dominates.
    ///
    /// # For Engineers
    ///
    /// This tells you how much RAM the model's learned parameters consume.
    /// It does NOT count the config struct or any temporary buffers used
    /// during computation.
    ///
    /// For the default config (dim=784): 784*784 + 784 = 615,440 floats.
    /// At 4 bytes per f32, that's about 2.4 MB -- well under the 10 MB
    /// budget for the "small" tier.
    ///
    /// Spec: SCENARIO-TIER-005
    pub fn parameter_memory_bytes(&self) -> usize {
        let float_size = std::mem::size_of::<Float>();
        // Coupling matrix: dim * dim floats
        let coupling_size = self.coupling.len() * float_size;
        // Bias vector: dim floats
        let bias_size = self.bias.len() * float_size;
        coupling_size + bias_size
    }
}

/// Implementation of the core `EnergyFunction` trait for the Ising model.
///
/// # For Researchers
///
/// Provides `energy(x) = -0.5 x^T J x - b^T x` and its exact gradient
/// `grad_energy(x) = -Jx - b`. No approximation or autograd needed.
///
/// # For Engineers
///
/// The `EnergyFunction` trait is the common interface that ALL Carnot models
/// implement. It requires three methods:
/// - `energy(x)`: Compute the scalar energy for a single input vector.
/// - `grad_energy(x)`: Compute the gradient of the energy with respect to x.
/// - `input_dim()`: Return the expected input dimensionality.
///
/// The trait also provides a default `energy_batch(xs)` that calls `energy()`
/// in a loop, which you get for free.
///
/// For engineers coming from PyTorch/TensorFlow: the `energy()` method is like
/// a forward pass, and `grad_energy()` is like calling `.backward()` -- except
/// here the gradient is computed with an explicit formula, not autograd.
impl EnergyFunction for IsingModel {
    /// Compute the energy for a single input vector x.
    ///
    /// # For Researchers
    ///
    /// E(x) = -0.5 * x^T J x - b^T x. O(dim^2) time.
    ///
    /// # For Engineers
    ///
    /// This is the core computation. Given an input vector x, it returns
    /// a single number (the energy). Here's what happens step by step:
    ///
    /// 1. `self.coupling.dot(x)` -- Matrix-vector multiply: J * x.
    ///    This produces a vector where entry i = sum_j(J[i][j] * x[j]),
    ///    i.e., the total "pull" that all other variables exert on variable i.
    ///
    /// 2. `x.dot(&(J*x))` -- Dot product: x^T * (J * x).
    ///    This produces a single scalar: the total pairwise interaction energy.
    ///
    /// 3. Multiply by -0.5 (the negative sign means low energy = strong agreement
    ///    between variables that have positive coupling).
    ///
    /// 4. `self.bias.dot(x)` -- The bias contribution: sum_i(b[i] * x[i]).
    ///    Subtract this to get the final energy.
    ///
    /// For example, with x = [1, -1], J = [[0, 2], [2, 0]], b = [0, 0]:
    ///   J*x = [2*(-1), 2*1] = [-2, 2]
    ///   x^T * J*x = 1*(-2) + (-1)*2 = -4
    ///   E = -0.5 * (-4) - 0 = 2.0
    /// The energy is positive because x[0] and x[1] have opposite signs
    /// but positive coupling (they "want" to agree but don't).
    ///
    /// Spec: REQ-CORE-001, SCENARIO-CORE-001
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        // Pairwise term: -0.5 * x^T * J * x
        // Bias term:     -b^T * x
        -0.5 * x.dot(&self.coupling.dot(x)) - self.bias.dot(x)
    }

    /// Compute the gradient of the energy with respect to the input x.
    ///
    /// # For Researchers
    ///
    /// dE/dx = -Jx - b (exact; exploits symmetry of J). O(dim^2) time.
    ///
    /// # For Engineers
    ///
    /// The gradient tells you: "if I nudge each element of x a tiny bit,
    /// how does the energy change?" This is crucial for sampling algorithms
    /// like Langevin dynamics, which use the gradient to find low-energy states.
    ///
    /// The math derivation (simplified):
    ///
    /// ```text
    /// E(x) = -0.5 * x^T J x - b^T x
    ///
    /// Taking the derivative with respect to x:
    ///   dE/dx = -0.5 * (J + J^T) * x - b
    ///
    /// But J is symmetric (J = J^T), so J + J^T = 2J:
    ///   dE/dx = -0.5 * 2J * x - b = -Jx - b
    /// ```
    ///
    /// For engineers coming from calculus: this is the same as the gradient
    /// of a quadratic form. If you remember that d/dx(x^T A x) = (A + A^T)x,
    /// then the -0.5 factor and symmetry of J simplify everything to -Jx.
    ///
    /// Spec: REQ-CORE-001, SCENARIO-CORE-003
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // Compute J * x: the "influence vector" -- how much each variable
        // is being pulled by all the others.
        let jx = self.coupling.dot(x);
        // Final gradient: negate the influence vector and subtract the bias.
        // The negative sign comes from the -0.5 in the energy formula
        // combined with the factor of 2 from differentiating x^T J x.
        -&jx - &self.bias
    }

    /// Return the expected input dimensionality.
    ///
    /// For engineers: use this to verify that your input vectors have the
    /// right length before calling `energy()` or `grad_energy()`.
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
