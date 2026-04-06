//! # carnot-boltzmann: Boltzmann (Large Tier) Energy-Based Model
//!
//! ## For Researchers
//!
//! Deep residual energy network with SiLU activations and optional skip projections.
//! Architecture: input projection -> N residual blocks -> linear readout -> scalar.
//! Analytical backpropagation through residual paths. Optional attention (planned).
//!
//! ## For Engineers Learning EBMs
//!
//! This is the most powerful model tier in Carnot, designed for research-scale
//! problems. While the Ising model is a single matrix multiply and the Gibbs model
//! is a plain MLP, the Boltzmann model uses **residual connections** -- the same
//! architecture innovation that made very deep networks (100+ layers) practical
//! in computer vision (ResNet) and language modeling (Transformers).
//!
//! ### What are residual connections and why do they matter?
//!
//! In a plain deep network, information must flow through every layer sequentially.
//! As the network gets deeper, gradients can "vanish" (shrink to near-zero) or
//! "explode" (grow uncontrollably). This makes training very deep plain networks
//! extremely difficult.
//!
//! A **residual connection** (also called a "skip connection") adds a shortcut:
//!
//! ```text
//! Plain layer:      output = f(input)
//! Residual layer:   output = f(input) + input     <-- the "+ input" is the skip
//! ```
//!
//! The "+input" means the layer only needs to learn the DIFFERENCE (residual)
//! between the input and the desired output, rather than the entire transformation.
//! This has two huge benefits:
//!
//! 1. **Gradient flow**: During backpropagation, gradients flow through BOTH
//!    the main path (through f) AND the skip path (identity). Even if the main
//!    path's gradients vanish, the skip path always passes gradients through
//!    unchanged. This is like having a highway lane that never gets blocked.
//!
//! 2. **Easy to learn identity**: If a layer doesn't need to do anything useful,
//!    it can just learn f(x) = 0, and the output is the input unchanged. In a
//!    plain network, the layer would have to learn the identity function explicitly.
//!
//! ### How each ResidualBlock works:
//!
//! ```text
//! input ----+---> [W1, b1] -> SiLU -> [W2, b2] -> SiLU ----> + ----> output
//!           |                                                 ^
//!           |              (skip/residual path)               |
//!           +---------[optional projection]-------------------+
//! ```
//!
//! - **Main path**: Two layers of linear transform + SiLU activation. This is
//!   where the "learning" happens -- the network detects and transforms patterns.
//!
//! - **Skip path**: The input is passed directly to the output (added to the
//!   main path's result). If the input and output dimensions differ, a linear
//!   projection matrix is applied to resize the input before adding.
//!
//! - **Output**: The sum of both paths. The network learns to adjust the main
//!   path to refine the input, rather than reconstruct it from scratch.
//!
//! ### Why backprop through residual blocks adds gradients from both paths
//!
//! During backpropagation, the chain rule tells us:
//!
//! ```text
//! output = main_path(input) + skip_path(input)
//!
//! d(output)/d(input) = d(main_path)/d(input) + d(skip_path)/d(input)
//! ```
//!
//! The gradient contributions from both paths are ADDED together. For the skip
//! path, d(skip)/d(input) is either the identity matrix (same dimensions) or
//! the transpose of the projection matrix (different dimensions). This means
//! the gradient through the skip path is always well-behaved -- it doesn't
//! pass through any activation functions that could squash it.
//!
//! ### Full model architecture:
//!
//! ```text
//! Input x (e.g., 784-dim)
//!   |
//!   v
//! [Input Projection] -- linear transform + SiLU to first hidden dim
//!   |
//!   v
//! [Residual Block 1] -- hidden_dims[0] -> hidden_dims[1]
//!   |
//!   v
//! [Residual Block 2] -- hidden_dims[1] -> hidden_dims[2]
//!   |
//!   v
//! ... (one block per adjacent pair of hidden dims) ...
//!   |
//!   v
//! [Output] -- dot product with output_weight + output_bias -> scalar energy
//! ```
//!
//! ### When to use the Boltzmann model
//!
//! - Research-scale problems where model expressiveness matters more than speed
//! - When simpler models (Ising, Gibbs) underfit your data
//! - When you need very deep networks (4+ layers) and gradient flow is critical
//! - Multi-modal energy landscapes with complex structure
//!
//! For engineers coming from deep learning: think of this as a ResNet, but
//! instead of outputting class probabilities, it outputs a scalar energy.
//! The residual blocks are conceptually identical to those in ResNet/ResNeXt.
//!
//! Spec: REQ-TIER-003, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006

use carnot_core::init::Initializer;
use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for the Boltzmann model.
///
/// # For Researchers
///
/// Configures the deep residual energy network: hidden layer widths,
/// number of attention heads (planned), residual toggle, and layer normalization.
///
/// # For Engineers
///
/// This struct defines the architecture of the most powerful Carnot model:
///
/// - `input_dim`: Size of the input vector (e.g., 784 for MNIST).
///
/// - `hidden_dims`: Sizes of hidden layers. Unlike Gibbs, these define the
///   dimensions of the RESIDUAL BLOCKS, not plain dense layers. Each adjacent
///   pair creates one residual block. For example, `vec![1024, 512, 256, 128]`
///   creates 3 residual blocks: 1024->512, 512->256, 256->128.
///
/// - `num_heads`: Number of attention heads (reserved for future multi-head
///   attention support). Must be > 0 for validation.
///
/// - `residual`: Whether to use residual (skip) connections. Set to `true` for
///   the standard architecture. Setting to `false` degrades to a plain deep
///   network (useful for ablation studies to measure the benefit of residuals).
///
/// - `layer_norm`: Whether to use layer normalization (reserved for future
///   implementation). Layer norm stabilizes training by normalizing hidden
///   representations.
///
/// For example:
/// ```rust
/// use carnot_boltzmann::BoltzmannConfig;
/// let config = BoltzmannConfig {
///     input_dim: 784,
///     hidden_dims: vec![512, 256, 128],
///     num_heads: 4,
///     residual: true,
///     layer_norm: true,
/// };
/// ```
///
/// Spec: REQ-TIER-005
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct BoltzmannConfig {
    /// Dimensionality of the input vector. Must be > 0.
    pub input_dim: usize,
    /// Sizes of hidden layers in order. Must have at least one entry.
    /// Each adjacent pair of dimensions creates one residual block.
    pub hidden_dims: Vec<usize>,
    /// Number of attention heads (reserved for future attention mechanism).
    /// Must be > 0.
    pub num_heads: usize,
    /// Whether to use residual (skip) connections in the residual blocks.
    /// When true: output = main_path(input) + skip(input).
    /// When false: output = main_path(input) (plain deep network).
    pub residual: bool,
    /// Whether to use layer normalization (reserved for future implementation).
    pub layer_norm: bool,
}

impl BoltzmannConfig {
    /// Validate that this configuration is well-formed.
    ///
    /// # For Engineers
    ///
    /// Checks: input_dim > 0, at least one hidden dim, all hidden dims > 0,
    /// num_heads > 0. Returns a descriptive error message on failure.
    ///
    /// Spec: SCENARIO-TIER-006
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
    /// Default config: 784 input, 4 hidden layers [1024, 512, 256, 128],
    /// 4 attention heads, residual connections ON, layer norm ON.
    ///
    /// For engineers: this creates a fairly deep and wide network suitable
    /// for research on MNIST-sized inputs. The decreasing hidden dims
    /// (1024 -> 512 -> 256 -> 128) create a "funnel" that progressively
    /// compresses the representation.
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

/// Compute the sigmoid function: 1 / (1 + exp(-x)).
///
/// For engineers: sigmoid squashes any real number into the range (0, 1).
/// Used internally by the SiLU activation function.
fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

/// SiLU (Sigmoid Linear Unit) activation: x * sigmoid(x).
///
/// # For Engineers
///
/// Also known as "Swish." This is a smooth, non-monotonic activation function
/// that has become the default in many modern architectures. Unlike ReLU, it
/// is differentiable everywhere and has no "dead zones."
///
/// Properties:
/// - silu(0) = 0
/// - For large positive x: silu(x) ~ x (behaves like identity)
/// - For large negative x: silu(x) ~ 0 (suppresses negative values)
/// - Has a small negative region near x ~ -1 (unlike ReLU)
fn silu(x: Float) -> Float {
    x * sigmoid(x)
}

/// Derivative of SiLU: sigmoid(x) * (1 + x * (1 - sigmoid(x))).
///
/// # For Engineers
///
/// This is needed during backpropagation. The derivative tells us how
/// sensitive the SiLU output is to changes in the input at each point.
///
/// Key property: silu_deriv is always positive for x > 0, and smoothly
/// transitions near zero. This smooth behavior helps gradient flow
/// compared to ReLU's hard cutoff at zero.
fn silu_deriv(x: Float) -> Float {
    let s = sigmoid(x);
    s * (1.0 + x * (1.0 - s))
}

/// A residual block: y = SiLU(W2 * SiLU(W1 * x + b1) + b2) + skip(x).
///
/// # For Researchers
///
/// Two-layer bottleneck with SiLU activations and optional linear projection
/// for dimension mismatch. Skip connection is identity when dims match.
///
/// # For Engineers
///
/// This is the core building block of the Boltzmann model. Each block has
/// TWO paths that process the input simultaneously:
///
/// **Main path** (the "learning" path):
/// ```text
/// x -> [W1 * x + b1] -> SiLU -> [W2 * a1 + b2] -> SiLU -> main_output
/// ```
/// This is a two-layer mini-network. The first layer (W1) can expand or
/// maintain the dimensionality, and the second layer (W2) maps to the
/// desired output dimension. Both layers use SiLU activation.
///
/// **Skip path** (the "shortcut" path):
/// ```text
/// x -> [optional projection P] -> skip_output
/// ```
/// If the input and output dimensions are the same, the skip path is just
/// the identity (pass x through unchanged). If dimensions differ, a linear
/// projection matrix P is used to resize x. There is no activation on the
/// skip path -- this is intentional! It ensures gradients flow through
/// unmodified, preventing vanishing gradients.
///
/// **Combined output**:
/// ```text
/// output = main_output + skip_output
/// ```
///
/// For example, a block mapping 512-dim to 256-dim:
/// - W1: [512, 512] (keeps dim), W2: [256, 512] (reduces dim)
/// - Projection P: [256, 512] (resizes skip path to match)
/// - output = SiLU(W2 * SiLU(W1*x + b1) + b2) + P*x
struct ResidualBlock {
    /// First layer weight matrix, shape [in_dim, in_dim].
    /// Maps input to intermediate representation (same dimensionality).
    w1: Array2<Float>,
    /// First layer bias vector, length in_dim.
    b1: Array1<Float>,
    /// Second layer weight matrix, shape [out_dim, in_dim].
    /// Maps intermediate representation to output dimensionality.
    w2: Array2<Float>,
    /// Second layer bias vector, length out_dim.
    b2: Array1<Float>,
    /// Optional projection matrix for the skip path, shape [out_dim, in_dim].
    /// Present only when in_dim != out_dim. When None, the skip path is identity.
    /// For engineers: this is how the "shortcut" handles dimension changes.
    /// Without it, you can't add a 512-dim input to a 256-dim main path output.
    proj: Option<Array2<Float>>,
    /// Whether to actually use the residual (skip) connection.
    /// When false, this block behaves like a plain two-layer network.
    use_residual: bool,
}

/// Cached intermediate values needed for backpropagation through a residual block.
///
/// # For Engineers
///
/// During the forward pass, we compute intermediate values (the "pre-activations")
/// that we'll need again during the backward pass to compute gradients. Rather
/// than recomputing them (wasteful) or storing the entire computation graph
/// (memory-heavy), we cache exactly what we need.
///
/// - `z1`: The values BEFORE the first SiLU activation (W1*x + b1).
///   Needed to compute SiLU'(z1) during backprop.
///
/// - `z2`: The values BEFORE the second SiLU activation (W2*a1 + b2).
///   Needed to compute SiLU'(z2) during backprop.
struct ResBlockCache {
    /// Pre-activation of the first SiLU: z1 = W1 * input + b1.
    z1: Array1<Float>,
    /// Pre-activation of the second SiLU: z2 = W2 * SiLU(z1) + b2.
    z2: Array1<Float>,
}

impl ResidualBlock {
    /// Forward pass: compute the block's output from the input.
    ///
    /// # For Engineers
    ///
    /// Computes: output = SiLU(W2 * SiLU(W1 * x + b1) + b2) + skip(x)
    ///
    /// Step by step:
    /// 1. z1 = W1 * x + b1 (first linear transform)
    /// 2. a1 = SiLU(z1)    (first activation)
    /// 3. z2 = W2 * a1 + b2 (second linear transform)
    /// 4. main_out = SiLU(z2) (second activation)
    /// 5. skip = proj * x (if dimensions differ) or x (if same)
    /// 6. output = main_out + skip (combine both paths)
    fn forward(&self, x: &Array1<Float>) -> Array1<Float> {
        // Main path: two layers of linear + SiLU
        let z1 = self.w1.dot(x) + &self.b1;
        let a1 = z1.mapv(silu); // Apply SiLU to first layer output
        let z2 = self.w2.dot(&a1) + &self.b2;
        let out = z2.mapv(silu); // Apply SiLU to second layer output

        if self.use_residual {
            // Skip path: identity or projection
            let skip = match &self.proj {
                Some(proj) => proj.dot(x), // Dimension change: apply projection
                None => x.clone(),         // Same dimensions: pass through directly
            };
            // Combine: element-wise addition of main path + skip path
            out + skip
        } else {
            // No residual: just return the main path output
            out
        }
    }

    /// Forward pass with caching for backpropagation.
    ///
    /// # For Engineers
    ///
    /// Identical to `forward()`, but also saves the pre-activation values z1 and z2
    /// in a `ResBlockCache`. These cached values are needed during the backward pass
    /// to compute the SiLU derivatives without redundant recomputation.
    ///
    /// Returns: (block_output, cache)
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

    /// Backward pass: given dE/d(output), compute dE/d(input).
    ///
    /// # For Researchers
    ///
    /// Backpropagates through both the main path (two SiLU layers) and the
    /// skip path (identity or projection). Gradients from both paths are summed.
    ///
    /// # For Engineers
    ///
    /// This is the heart of gradient computation for deep residual networks.
    /// Given how much the energy changes with respect to THIS block's output
    /// (d_out), we compute how much it changes with respect to THIS block's INPUT.
    ///
    /// The gradient flows through TWO paths and the results are ADDED:
    ///
    /// **Main path gradient** (through the two SiLU layers):
    /// ```text
    /// d_out
    ///   -> multiply by SiLU'(z2)         = d_silu2   (gradient through 2nd activation)
    ///   -> multiply by W2^T              = d_a1      (gradient through 2nd linear)
    ///   -> multiply by SiLU'(z1)         = d_silu1   (gradient through 1st activation)
    ///   -> multiply by W1^T              = d_input_main (gradient through 1st linear)
    /// ```
    ///
    /// **Skip path gradient** (through identity or projection):
    /// ```text
    /// d_out
    ///   -> multiply by P^T (if projection exists) or pass through as-is
    ///   = d_input_skip
    /// ```
    ///
    /// **Total gradient**: d_input = d_input_main + d_input_skip
    ///
    /// The key insight: even if the main path's gradient vanishes (e.g., SiLU
    /// derivatives are near zero), the skip path gradient ALWAYS flows through.
    /// With identity skip: d_input_skip = d_out (gradient passes unchanged).
    /// With projection: d_input_skip = P^T * d_out (just a linear transform).
    /// Neither path involves activation functions, so no squashing occurs.
    fn backward(&self, d_out: &Array1<Float>, cache: &ResBlockCache) -> Array1<Float> {
        // === Gradient through the MAIN path ===

        // Step 1: Gradient through second SiLU.
        // output = silu(z2) + skip, so d(silu(z2))/d(z2) = silu'(z2).
        // d_silu2 = d_out * silu'(z2) (elementwise).
        let d_silu2 = d_out * &cache.z2.mapv(silu_deriv);

        // Step 2: Gradient through second linear layer.
        // z2 = W2 * a1 + b2, so dE/d(a1) = W2^T * dE/d(z2).
        let d_a1 = self.w2.t().dot(&d_silu2);

        // Step 3: Gradient through first SiLU.
        // a1 = silu(z1), so dE/d(z1) = dE/d(a1) * silu'(z1) (elementwise).
        let d_silu1 = &d_a1 * &cache.z1.mapv(silu_deriv);

        // Step 4: Gradient through first linear layer.
        // z1 = W1 * x + b1, so dE/d(x) via main path = W1^T * dE/d(z1).
        let mut d_input = self.w1.t().dot(&d_silu1);

        // === Gradient through the SKIP path ===
        // The skip path gradient is ADDED to the main path gradient.
        // This is the key benefit of residual connections: the gradient
        // has an "express lane" that bypasses all the nonlinear layers.
        if self.use_residual {
            match &self.proj {
                Some(proj) => {
                    // Skip uses a projection: skip = P * x.
                    // So dE/d(x) via skip = P^T * d_out.
                    d_input += &proj.t().dot(d_out);
                }
                None => {
                    // Skip is identity: skip = x.
                    // So dE/d(x) via skip = d_out (gradient passes through unchanged!).
                    // This is the simplest and most powerful case: no matter what
                    // happens in the main path, the gradient from d_out reaches
                    // d_input without any modification.
                    d_input += d_out;
                }
            }
        }

        d_input
    }
}

/// The Boltzmann Energy-Based Model.
///
/// # For Researchers
///
/// Deep residual energy network. Architecture:
/// input -> linear projection + SiLU -> N residual blocks -> linear readout -> scalar.
/// Analytical gradient via chain rule through all residual paths.
/// Parameter count is dominated by the first hidden layer (input_dim * hidden_dims[0]).
///
/// # For Engineers
///
/// This is the flagship model of Carnot. It combines:
///
/// - **Input projection**: A linear transform + SiLU that maps the raw input
///   into the first hidden dimension. This is the "entry point" into the
///   residual network.
///
/// - **Residual blocks**: A stack of blocks, each containing two SiLU layers
///   plus a skip connection. These do the heavy lifting of feature extraction.
///
/// - **Output readout**: A simple dot product that collapses the last hidden
///   representation into a scalar energy value.
///
/// Memory and compute scale with the hidden dimensions. The default config
/// (784 -> [1024, 512, 256, 128]) has roughly ~1.5M parameters. This is
/// small by modern deep learning standards but large enough for meaningful
/// energy landscape modeling.
///
/// For engineers coming from PyTorch: this is conceptually equivalent to:
/// ```python
/// class BoltzmannModel(nn.Module):
///     def __init__(self):
///         self.input_proj = nn.Linear(784, 1024)
///         self.blocks = nn.ModuleList([ResidualBlock(...), ...])
///         self.output = nn.Linear(128, 1)
///     def forward(self, x):
///         h = F.silu(self.input_proj(x))
///         for block in self.blocks:
///             h = block(h)
///         return self.output(h)  # scalar energy
/// ```
///
/// Spec: REQ-TIER-003
pub struct BoltzmannModel {
    /// Input projection matrix, shape [hidden_dims[0], input_dim].
    /// Maps raw input into the first hidden dimension.
    input_proj: Array2<Float>,
    /// Input projection bias, length hidden_dims[0].
    input_bias: Array1<Float>,
    /// Stack of residual blocks, one per adjacent pair of hidden dimensions.
    /// For hidden_dims = [1024, 512, 256, 128], there are 3 blocks:
    ///   Block 0: 1024 -> 512
    ///   Block 1: 512 -> 256
    ///   Block 2: 256 -> 128
    blocks: Vec<ResidualBlock>,
    /// Output weight vector, length = last hidden dim.
    /// Dot product with the final hidden representation to produce scalar energy.
    output_weight: Array1<Float>,
    /// Scalar bias added to the final energy.
    output_bias: Float,
    /// The configuration used to create this model.
    config: BoltzmannConfig,
}

impl BoltzmannModel {
    /// Create a new Boltzmann model with the given configuration.
    ///
    /// # For Researchers
    ///
    /// Initializes all weights with Xavier uniform. Input projection, residual
    /// block weights, and optional skip projections are all Xavier-initialized.
    /// Biases and output weights are zero-initialized.
    ///
    /// # For Engineers
    ///
    /// This constructor builds the deep residual network:
    ///
    /// 1. **Input projection**: Creates a weight matrix mapping input_dim to
    ///    hidden_dims[0], plus a bias vector. This is the "entry ramp" into
    ///    the residual network.
    ///
    /// 2. **Residual blocks**: For each pair of adjacent hidden dimensions
    ///    (e.g., 1024->512), creates a ResidualBlock with:
    ///    - W1: [in_dim, in_dim] (first layer, keeps dimensionality)
    ///    - W2: [out_dim, in_dim] (second layer, may change dimensionality)
    ///    - Projection P: [out_dim, in_dim] (only if in_dim != out_dim)
    ///
    /// 3. **Output layer**: A weight vector (length = last hidden dim) and
    ///    scalar bias. Zero-initialized so initial energy = 0 for all inputs.
    ///
    /// For example, with hidden_dims = [1024, 512, 256]:
    ///   - Input proj: [1024, 784]
    ///   - Block 0: W1=[1024,1024], W2=[512,1024], proj=[512,1024]
    ///   - Block 1: W1=[512,512], W2=[256,512], proj=[256,512]
    ///   - Output: w=[256], b=0.0
    ///
    /// Spec: REQ-TIER-003, REQ-TIER-005, SCENARIO-TIER-006
    pub fn new(config: BoltzmannConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        let init = Initializer::XavierUniform;

        // Input projection: maps raw input into the network's hidden space.
        let input_proj = init.init_matrix(config.hidden_dims[0], config.input_dim);
        let input_bias = Array1::zeros(config.hidden_dims[0]);

        // Build residual blocks for each adjacent pair of hidden dimensions.
        let mut blocks = Vec::new();
        for i in 0..config.hidden_dims.len() - 1 {
            let in_dim = config.hidden_dims[i];
            let out_dim = config.hidden_dims[i + 1];

            // First layer: [in_dim, in_dim] -- processes within the same dimension.
            let w1 = init.init_matrix(in_dim, in_dim);
            let b1 = Array1::zeros(in_dim);

            // Second layer: [out_dim, in_dim] -- may change dimension.
            let w2 = init.init_matrix(out_dim, in_dim);
            let b2 = Array1::zeros(out_dim);

            // Skip projection: only needed when dimensions change.
            // If in_dim == out_dim, the skip path is identity (no projection needed).
            // If in_dim != out_dim, we need a linear projection to resize the
            // skip connection so it can be added to the main path's output.
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

        // Output layer: collapses the final hidden representation to a scalar.
        // Zero-initialized so the model starts with E(x) = 0 for all x.
        let last_dim = *config.hidden_dims.last().ok_or_else(|| {
            CarnotError::InvalidConfig("hidden_dims must have at least one layer".into())
        })?;
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

/// Implementation of the core `EnergyFunction` trait for the Boltzmann model.
///
/// # For Engineers
///
/// The Boltzmann model implements the same trait as Ising and Gibbs, so you can
/// swap models without changing your sampling or training code. The key differences:
///
/// - Ising: quadratic energy (no hidden layers, O(dim^2))
/// - Gibbs: MLP energy (plain hidden layers, moderate depth)
/// - Boltzmann: residual network energy (skip connections, deepest and most expressive)
///
/// The gradient computation uses analytical backpropagation through the entire
/// residual architecture, including both the main and skip paths of each block.
impl EnergyFunction for BoltzmannModel {
    /// Compute the scalar energy for a single input vector x.
    ///
    /// # For Researchers
    ///
    /// E(x) = w^T * ResBlock_N(...ResBlock_1(SiLU(Proj * x + b_proj))...) + b_out
    ///
    /// # For Engineers
    ///
    /// The forward pass:
    ///
    /// 1. **Input projection**: h = SiLU(input_proj * x + input_bias)
    ///    Maps the raw input into the hidden space and applies a nonlinearity.
    ///
    /// 2. **Residual blocks**: h = block_N(... block_1(h) ...)
    ///    Each block refines the representation through its main path + skip.
    ///
    /// 3. **Output readout**: E = output_weight^T * h + output_bias
    ///    Collapses the final hidden vector to a single scalar energy.
    ///
    /// Spec: REQ-CORE-001, SCENARIO-TIER-003
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        // Step 1: Project input into hidden space and apply SiLU.
        let mut h = self.input_proj.dot(x) + &self.input_bias;
        h = h.mapv(silu);

        // Step 2: Pass through each residual block sequentially.
        // Each block refines the representation, with skip connections
        // ensuring gradient flow and learning stability.
        for block in &self.blocks {
            h = block.forward(&h);
        }

        // Step 3: Collapse to scalar energy via dot product.
        self.output_weight.dot(&h) + self.output_bias
    }

    /// Compute the gradient of the energy with respect to the input x
    /// using analytical backpropagation through all residual blocks.
    ///
    /// # For Researchers
    ///
    /// Full chain rule through input projection + SiLU, N residual blocks
    /// (each with main + skip paths), and linear readout. O(sum of block sizes).
    ///
    /// # For Engineers
    ///
    /// This implements backpropagation through the entire deep residual network.
    /// The gradient flows backward through each component:
    ///
    /// ```text
    /// Forward:  x -> [input_proj + SiLU] -> [Block 1] -> [Block 2] -> ... -> [readout] -> E
    /// Backward: dE/dx <- [input_proj^T * ...] <- [Block 1 backward] <- [Block 2 backward] <- ... <- dE/dh = w_out
    /// ```
    ///
    /// Key insight for the residual blocks: during backward, each block's
    /// `backward()` method sums gradients from BOTH the main path and the
    /// skip path. This is why residual networks train well even when very deep:
    /// the skip path gradients always reach the input, even if the main path
    /// gradients vanish through the SiLU activations.
    ///
    /// The overall cost is one forward pass (to cache intermediate values)
    /// plus one backward pass. This is O(1) forward passes regardless of
    /// input dimensionality, compared to O(2 * input_dim) for finite differences.
    ///
    /// Spec: REQ-CORE-001, SCENARIO-CORE-003
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // === Forward pass with caching ===
        // We need to save intermediate values for the backward pass.

        // Input projection: z_input = input_proj * x + input_bias.
        // We cache z_input (pre-SiLU) because we need silu_deriv(z_input) later.
        let z_input = self.input_proj.dot(x) + &self.input_bias;
        let mut h = z_input.mapv(silu);

        // Forward through each residual block, caching pre-activation values.
        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (output, cache) = block.forward_with_cache(&h);
            block_caches.push(cache);
            h = output;
        }

        // === Backward pass ===

        // Start at the output: E = w^T * h_final + b, so dE/dh_final = w.
        let mut delta = self.output_weight.clone();

        // Backprop through residual blocks in reverse order.
        // Each block.backward() returns dE/d(block_input) given dE/d(block_output).
        // Crucially, this includes gradients from BOTH the main and skip paths.
        for i in (0..self.blocks.len()).rev() {
            delta = self.blocks[i].backward(&delta, &block_caches[i]);
        }

        // Backprop through the input SiLU activation.
        // h_0 = silu(z_input), so dE/d(z_input) = delta * silu'(z_input).
        let d_z_input = &delta * &z_input.mapv(silu_deriv);

        // Backprop through the input projection.
        // z_input = input_proj * x + input_bias, so dE/dx = input_proj^T * dE/d(z_input).
        self.input_proj.t().dot(&d_z_input)
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
