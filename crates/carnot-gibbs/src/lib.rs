//! # carnot-gibbs: Gibbs (Medium Tier) Energy-Based Model
//!
//! ## For Researchers
//!
//! Multi-layer energy network: E(x) = w^T f_L(...f_2(f_1(x))) + b -> scalar.
//! Supports SiLU, ReLU, and Tanh activations. Gradients are computed via
//! analytical backpropagation (no autograd or finite differences).
//!
//! ## For Engineers Learning EBMs
//!
//! If the Ising model is a "Hello World" EBM (just a matrix multiply), the
//! Gibbs model is a proper neural network -- except instead of outputting class
//! probabilities, it outputs a single scalar energy value.
//!
//! ### How it works at a high level:
//!
//! ```text
//! Input x (e.g., 784-dim image)
//!   |
//!   v
//! [Dense Layer 1] -- weight matrix W1, bias b1, activation function
//!   |
//!   v
//! [Dense Layer 2] -- weight matrix W2, bias b2, activation function
//!   |
//!   v
//! ... (as many layers as you configure) ...
//!   |
//!   v
//! [Output] -- a single dot product: w_out^T * last_hidden + b_out
//!   |
//!   v
//! Scalar energy E(x)
//! ```
//!
//! Each "Dense Layer" does: output = activation(W * input + b).
//! The final output layer has NO activation -- it just linearly combines the
//! last hidden layer into a single number.
//!
//! ### What "analytical backpropagation" means and why it matters
//!
//! To train or sample from an EBM, we need the gradient dE/dx -- how the
//! energy changes when we tweak the input. There are three ways to get this:
//!
//! 1. **Finite differences**: Nudge each input dimension by a tiny epsilon and
//!    measure the energy change. Simple but SLOW (requires 2*dim forward passes)
//!    and numerically imprecise.
//!
//! 2. **Autograd** (like PyTorch's `.backward()`): A library records all
//!    operations and automatically computes gradients. Convenient but adds
//!    overhead from the computation graph.
//!
//! 3. **Analytical backpropagation** (what we do here): We derive the gradient
//!    formula by hand using the chain rule, then implement it directly. This is
//!    the fastest and most precise approach, but requires manual derivation for
//!    each architecture. Since the Gibbs model has a fixed architecture (stack
//!    of dense layers), we can do this once and it works for any configuration.
//!
//! ### What the activation functions do
//!
//! Without activation functions, stacking layers would be pointless -- multiple
//! linear transforms collapse into one. Activations add nonlinearity so the
//! network can learn complex energy landscapes.
//!
//! - **SiLU** (Sigmoid Linear Unit, aka Swish): `x * sigmoid(x)`. Smooth,
//!   differentiable everywhere. The default choice. Good for most problems.
//!   Unlike ReLU, it has a smooth curve near zero, which helps optimization.
//!
//! - **ReLU** (Rectified Linear Unit): `max(0, x)`. The classic. Simple and
//!   fast, but has a "dead zone" where x < 0 (gradient is exactly 0). Can
//!   cause "dead neurons" that never activate.
//!
//! - **Tanh** (Hyperbolic Tangent): `tanh(x)`. Outputs are bounded to (-1, 1).
//!   Useful when you want to constrain hidden values. Can suffer from vanishing
//!   gradients for large inputs.
//!
//! For engineers coming from classification networks: the architecture is almost
//! identical to a standard MLP classifier, except:
//! (a) the final layer outputs a scalar (not a class vector), and
//! (b) there is no softmax -- the raw scalar IS the energy.
//!
//! Spec: REQ-TIER-002, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006

use carnot_core::init::Initializer;
use carnot_core::{CarnotError, EnergyFunction, Float};
use ndarray::{Array1, Array2, ArrayView1};

/// Configuration for the Gibbs model.
///
/// # For Researchers
///
/// Specifies the network topology (layer widths), activation function, and
/// dropout rate.
///
/// # For Engineers
///
/// This struct defines the shape and behavior of your neural network energy model:
///
/// - `input_dim`: Size of the input vector (e.g., 784 for MNIST images).
///
/// - `hidden_dims`: A list of hidden layer sizes. For example, `vec![512, 256]`
///   means "first hidden layer has 512 neurons, second has 256." The network
///   progressively compresses the representation. More/wider layers = more
///   expressive but slower and more memory.
///
/// - `activation`: Which nonlinear function to apply after each hidden layer.
///   See the module-level docs for guidance on SiLU vs ReLU vs Tanh.
///
/// - `dropout`: Fraction of neurons to randomly zero out during training
///   (regularization). Must be in [0, 1). Set to 0.0 to disable.
///   NOTE: Dropout is not currently applied during inference -- this parameter
///   is reserved for future training support.
///
/// For example:
/// ```rust
/// use carnot_gibbs::{GibbsConfig, Activation};
/// let config = GibbsConfig {
///     input_dim: 100,
///     hidden_dims: vec![64, 32],
///     activation: Activation::SiLU,
///     dropout: 0.0,
/// };
/// ```
///
/// Spec: REQ-TIER-005
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GibbsConfig {
    /// Dimensionality of the input vector. Must be > 0.
    pub input_dim: usize,
    /// Sizes of hidden layers, in order from input to output.
    /// Must have at least one entry, and all entries must be > 0.
    ///
    /// For example: `vec![512, 256]` creates two hidden layers.
    /// The network shape is: input_dim -> 512 -> 256 -> 1 (scalar energy).
    pub hidden_dims: Vec<usize>,
    /// Which activation function to use in every hidden layer.
    pub activation: Activation,
    /// Dropout probability in [0, 1). Reserved for future training support.
    pub dropout: f64,
}

/// Available activation functions for the Gibbs model.
///
/// # For Researchers
///
/// SiLU, ReLU, and Tanh -- standard choices for MLP energy functions.
///
/// # For Engineers
///
/// An activation function is a simple mathematical function applied to each
/// neuron's output. It adds "nonlinearity" -- without it, no matter how many
/// layers you stack, the whole network would just be one big matrix multiply.
///
/// - **SiLU**: Smooth, no dead zones. Best default. `f(x) = x * sigmoid(x)`
/// - **ReLU**: Fast, sparse activations. `f(x) = max(0, x)`
/// - **Tanh**: Bounded output in (-1,1). `f(x) = tanh(x)`
///
/// When to use which:
/// - Start with SiLU (the default). It works well for most energy landscapes.
/// - Use ReLU if you need maximum speed and don't mind potential dead neurons.
/// - Use Tanh if your problem benefits from bounded hidden representations.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum Activation {
    /// Sigmoid Linear Unit: x * sigmoid(x). Smooth everywhere.
    /// Derivative: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    SiLU,
    /// Rectified Linear Unit: max(0, x). Simple and fast.
    /// Derivative: 1 if x > 0, else 0 (undefined at 0, we use 0).
    ReLU,
    /// Hyperbolic tangent: tanh(x). Bounded to (-1, 1).
    /// Derivative: 1 - tanh(x)^2
    Tanh,
}

impl GibbsConfig {
    /// Validate that this configuration is well-formed.
    ///
    /// # For Engineers
    ///
    /// Checks:
    /// - `input_dim > 0`
    /// - `hidden_dims` is not empty
    /// - All hidden dimensions are > 0
    /// - `dropout` is in [0, 1)
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
        if !(0.0..1.0).contains(&self.dropout) {
            return Err(CarnotError::InvalidConfig(
                "dropout must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }
}

impl Default for GibbsConfig {
    /// Default config: 784 input dim, two hidden layers [512, 256], SiLU activation, no dropout.
    ///
    /// For engineers: this is sized for MNIST-like data. The two-layer architecture
    /// with decreasing widths is a common "funnel" pattern that progressively
    /// compresses the representation before producing a scalar energy.
    fn default() -> Self {
        Self {
            input_dim: 784,
            hidden_dims: vec![512, 256],
            activation: Activation::SiLU,
            dropout: 0.0,
        }
    }
}

/// A single dense (fully connected) layer: y = activation(W * x + b).
///
/// # For Engineers
///
/// This is the basic building block of the Gibbs model. Each dense layer:
///
/// 1. Multiplies the input by a weight matrix W (a linear transformation).
/// 2. Adds a bias vector b (shifts the result).
/// 3. Applies an activation function (adds nonlinearity).
///
/// The weight matrix W has shape [output_dim, input_dim], so it transforms
/// a vector of size `input_dim` into one of size `output_dim`.
///
/// For example, a layer with W shape [256, 512] takes a 512-dim input and
/// produces a 256-dim output, compressing the representation.
struct DenseLayer {
    /// Weight matrix W, shape [output_dim, input_dim].
    /// Each row is a "detector" that looks for a particular pattern in the input.
    weight: Array2<Float>,
    /// Bias vector b, length output_dim.
    /// Shifts the pre-activation values so neurons can activate even when
    /// the input doesn't perfectly match the weight pattern.
    bias: Array1<Float>,
    /// Which activation function to apply after the linear transform.
    activation: Activation,
}

/// Compute the sigmoid function: 1 / (1 + exp(-x)).
///
/// For engineers: sigmoid maps any real number to the range (0, 1).
/// It's used internally by the SiLU activation, not as a standalone activation.
fn sigmoid(x: Float) -> Float {
    1.0 / (1.0 + (-x).exp())
}

/// Apply an activation function elementwise to a vector.
///
/// # For Engineers
///
/// Given a vector of "pre-activation" values z (the raw output of W*x + b),
/// this function applies the chosen nonlinearity to each element independently.
///
/// - SiLU: each element becomes `z_i * sigmoid(z_i)` -- smooth S-curve times z.
/// - ReLU: each element becomes `max(0, z_i)` -- zeroes out negatives.
/// - Tanh: each element becomes `tanh(z_i)` -- squashes to (-1, 1).
fn activate(z: &Array1<Float>, act: Activation) -> Array1<Float> {
    match act {
        Activation::SiLU => z.mapv(|v| v * sigmoid(v)),
        Activation::ReLU => z.mapv(|v| v.max(0.0)),
        Activation::Tanh => z.mapv(|v| v.tanh()),
    }
}

/// Compute the derivative of the activation function at each pre-activation value z.
///
/// # For Researchers
///
/// Returns d(activation(z))/dz elementwise. Used in analytical backpropagation.
///
/// # For Engineers
///
/// During backpropagation, we need to know: "if I change the pre-activation z
/// by a tiny amount, how much does the activation output change?" That rate of
/// change is the derivative, and it's different for each activation function:
///
/// - **SiLU derivative**: `sigmoid(z) * (1 + z * (1 - sigmoid(z)))`.
///   This is always positive, and smoothly varies -- no sudden jumps.
///
/// - **ReLU derivative**: `1` if z > 0, `0` if z <= 0.
///   This is the "all or nothing" nature of ReLU -- gradients either flow
///   through completely or are blocked entirely.
///
/// - **Tanh derivative**: `1 - tanh(z)^2`.
///   Largest near z=0 (value = 1), shrinks toward 0 for large |z|.
///   This is why Tanh can suffer from "vanishing gradients" in deep networks.
fn activate_deriv(z: &Array1<Float>, act: Activation) -> Array1<Float> {
    match act {
        Activation::SiLU => {
            // d/dz [z * sigmoid(z)] = sigmoid(z) + z * sigmoid(z) * (1 - sigmoid(z))
            //                        = sigmoid(z) * (1 + z * (1 - sigmoid(z)))
            z.mapv(|v| {
                let s = sigmoid(v);
                s * (1.0 + v * (1.0 - s))
            })
        }
        Activation::ReLU => z.mapv(|v| if v > 0.0 { 1.0 } else { 0.0 }),
        Activation::Tanh => {
            // d/dz [tanh(z)] = 1 - tanh(z)^2
            z.mapv(|v| {
                let t = v.tanh();
                1.0 - t * t
            })
        }
    }
}

impl DenseLayer {
    /// Forward pass: compute output = activation(W * x + b).
    ///
    /// For engineers: this is the standard "forward propagation" step.
    /// Input x flows through the linear transform (W*x + b), then through
    /// the nonlinear activation function.
    fn forward(&self, x: &Array1<Float>) -> Array1<Float> {
        // Step 1: Linear transform -- z = W * x + b
        let z = self.weight.dot(x) + &self.bias;
        // Step 2: Nonlinear activation -- output = activation(z)
        activate(&z, self.activation)
    }

    /// Forward pass that also returns the pre-activation values z.
    ///
    /// # For Engineers
    ///
    /// This is identical to `forward()` but also saves the intermediate value
    /// z = W*x + b (before the activation function is applied). We need z later
    /// for backpropagation -- specifically, to compute the activation derivative.
    ///
    /// Returns: (activated_output, pre_activation_z)
    ///
    /// Why cache z? During backprop, we compute activation'(z) -- the derivative
    /// of the activation at the pre-activation values. Without caching z, we'd
    /// have to recompute W*x + b during the backward pass, wasting time.
    fn forward_with_cache(&self, x: &Array1<Float>) -> (Array1<Float>, Array1<Float>) {
        let z = self.weight.dot(x) + &self.bias;
        let a = activate(&z, self.activation);
        (a, z)
    }
}

/// The Gibbs Energy-Based Model.
///
/// # For Researchers
///
/// Multi-layer perceptron energy function: E(x) = w^T f(x) + b, where f(x)
/// is the composition of L dense layers with configurable activations.
/// Analytical gradient via chain rule backpropagation.
///
/// # For Engineers
///
/// This is a neural network that outputs energy. It contains:
///
/// - **`layers`**: A stack of dense layers that transform the input step by step.
///   Each layer reduces (or changes) the dimensionality. For example, with
///   hidden_dims = [512, 256], the first layer maps input_dim -> 512, and
///   the second maps 512 -> 256.
///
/// - **`output_weight`** and **`output_bias`**: The final layer that collapses
///   the last hidden representation into a single scalar energy value.
///   E = output_weight^T * last_hidden + output_bias.
///
/// The model architecture looks like a funnel:
///
/// ```text
/// wide input (e.g., 784) -> narrower hidden -> narrower hidden -> scalar
/// ```
///
/// For engineers coming from classification: imagine a classifier that outputs
/// a single number instead of class probabilities. That number is the energy.
/// Low energy = "this input looks natural/likely." High energy = "this input
/// looks weird/unlikely."
///
/// Spec: REQ-TIER-002
pub struct GibbsModel {
    /// Stack of dense hidden layers, in order from input to output.
    layers: Vec<DenseLayer>,
    /// Weight vector for the final linear combination, length = last hidden dim.
    /// This dot-products with the last hidden layer to produce the scalar energy.
    output_weight: Array1<Float>,
    /// Scalar bias added to the final energy output.
    output_bias: Float,
    /// The configuration used to create this model.
    config: GibbsConfig,
}

impl GibbsModel {
    /// Create a new Gibbs model with the given configuration.
    ///
    /// # For Researchers
    ///
    /// Initializes all weight matrices with Xavier uniform. Biases are zero-initialized.
    /// Output layer (w, b) is zero-initialized (energy starts at 0 for all inputs).
    ///
    /// # For Engineers
    ///
    /// This constructor builds the neural network layer by layer:
    ///
    /// 1. For each pair of consecutive dimensions (input_dim -> hidden_dims[0],
    ///    hidden_dims[0] -> hidden_dims[1], etc.), it creates a DenseLayer with
    ///    Xavier-initialized weights and zero biases.
    ///
    /// 2. The final output layer is a simple dot product (no activation), mapping
    ///    the last hidden dimension to a scalar. It starts at zero so the model
    ///    initially assigns energy 0 to all inputs.
    ///
    /// For example, with input_dim=784 and hidden_dims=[512, 256]:
    /// - Layer 1: W shape [512, 784], b shape [512]
    /// - Layer 2: W shape [256, 512], b shape [256]
    /// - Output: w shape [256], b = 0.0
    ///
    /// Spec: REQ-TIER-002, REQ-TIER-005, SCENARIO-TIER-006
    pub fn new(config: GibbsConfig) -> Result<Self, CarnotError> {
        config.validate()?;

        let init = Initializer::XavierUniform;
        let mut layers = Vec::new();

        // Build hidden layers: each layer transforms from prev_dim to hidden_dim.
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

        // Output layer: a simple linear combination (dot product) from the last
        // hidden dimension to a scalar. Zero-initialized so initial energy = 0.
        let output_weight = Array1::zeros(prev_dim);
        let output_bias = 0.0;

        Ok(Self {
            layers,
            output_weight,
            output_bias,
            config,
        })
    }

    /// Run the input through all hidden layers, returning the final hidden representation.
    ///
    /// For engineers: this is the "feature extraction" part. The input is transformed
    /// layer by layer, with each layer applying a linear transform followed by a
    /// nonlinear activation. The result is a compact representation that the output
    /// layer will convert to a scalar energy.
    fn forward_hidden(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        let mut h = x.to_owned();
        for layer in &self.layers {
            h = layer.forward(&h);
        }
        h
    }
}

/// Implementation of the core `EnergyFunction` trait for the Gibbs model.
///
/// # For Engineers
///
/// This is where the magic happens. The Gibbs model implements the same
/// trait as the simpler Ising model, so you can swap between them without
/// changing your sampling or training code. The key difference:
///
/// - Ising: energy is a quadratic function (no hidden layers)
/// - Gibbs: energy comes from a multi-layer neural network (much more expressive)
///
/// The gradient computation uses analytical backpropagation -- the same algorithm
/// as PyTorch's autograd, but implemented by hand for maximum performance.
impl EnergyFunction for GibbsModel {
    /// Compute the scalar energy for a single input vector x.
    ///
    /// # For Researchers
    ///
    /// E(x) = w^T f_L(W_L f_{L-1}(...f_1(W_1 x + b_1)...) + b_L) + b_out
    ///
    /// # For Engineers
    ///
    /// The forward pass works like this:
    ///
    /// 1. Pass x through each hidden layer in sequence. Each layer applies:
    ///    h_next = activation(W * h_current + b)
    ///
    /// 2. Take the final hidden vector and dot-product it with the output weight
    ///    vector, then add the output bias. This produces a single scalar: the energy.
    ///
    /// For example, with 2 hidden layers:
    ///   h1 = SiLU(W1 * x + b1)        -- 784-dim -> 512-dim
    ///   h2 = SiLU(W2 * h1 + b2)       -- 512-dim -> 256-dim
    ///   E  = w_out^T * h2 + b_out      -- 256-dim -> scalar
    ///
    /// Spec: REQ-CORE-001, SCENARIO-TIER-002
    fn energy(&self, x: &ArrayView1<Float>) -> Float {
        // Run input through all hidden layers to get the final representation.
        let h = self.forward_hidden(x);
        // Collapse to scalar: dot product with output weights + bias.
        self.output_weight.dot(&h) + self.output_bias
    }

    /// Compute the gradient of the energy with respect to the input x
    /// using analytical backpropagation.
    ///
    /// # For Researchers
    ///
    /// E(x) = w^T * f_L(...f_1(x)) + b
    /// dE/dx = dE/dh_L * dh_L/dh_{L-1} * ... * dh_1/dx
    /// Each Jacobian factor is W_i^T * diag(activation'(z_i)).
    ///
    /// # For Engineers
    ///
    /// Backpropagation computes the gradient by working BACKWARDS through the
    /// network, using the chain rule from calculus. Here's the intuition:
    ///
    /// **Forward pass** (we cache intermediate values):
    /// ```text
    /// x -> [Layer 1] -> h1 (and cache z1 = W1*x + b1)
    ///              -> [Layer 2] -> h2 (and cache z2 = W2*h1 + b2)
    ///                         -> [Output] -> E = w^T * h2 + b
    /// ```
    ///
    /// **Backward pass** (compute gradients right to left):
    /// ```text
    /// dE/dh2 = w_out                          (trivial: E = w^T * h2 + b)
    /// dE/dz2 = dE/dh2 * activation'(z2)       (chain rule through activation)
    /// dE/dh1 = W2^T * dE/dz2                  (chain rule through linear transform)
    /// dE/dz1 = dE/dh1 * activation'(z1)       (chain rule through activation)
    /// dE/dx  = W1^T * dE/dz1                  (chain rule through linear transform)
    /// ```
    ///
    /// The * in "dE/dh2 * activation'(z2)" is elementwise multiplication (Hadamard
    /// product), because the activation is applied independently to each element.
    ///
    /// Why is this better than finite differences?
    /// - Finite differences need 2*dim forward passes (one +epsilon, one -epsilon
    ///   for each dimension). For dim=784, that's 1568 forward passes.
    /// - Backprop needs exactly ONE forward pass + ONE backward pass, regardless
    ///   of dimensionality. That's orders of magnitude faster.
    ///
    /// Spec: REQ-CORE-001, SCENARIO-CORE-003
    fn grad_energy(&self, x: &ArrayView1<Float>) -> Array1<Float> {
        // === Forward pass: cache all intermediate values ===
        // We need both the activations (for layer inputs) and pre-activations
        // (for computing activation derivatives during backprop).
        let mut activations = Vec::with_capacity(self.layers.len() + 1);
        let mut pre_activations = Vec::with_capacity(self.layers.len());

        let mut h = x.to_owned();
        activations.push(h.clone()); // Save the original input as "activation 0"

        for layer in &self.layers {
            // forward_with_cache returns (activated_output, pre_activation_z)
            let (a, z) = layer.forward_with_cache(&h);
            pre_activations.push(z); // Cache z for backprop (need it for activation')
            activations.push(a.clone()); // Cache activated output (input to next layer)
            h = a;
        }

        // === Backward pass: compute gradient layer by layer in reverse ===

        // Start from the output: E = w^T * h_final + b, so dE/dh_final = w.
        let mut delta = self.output_weight.clone();

        // Walk backwards through the layers, propagating the gradient.
        for i in (0..self.layers.len()).rev() {
            // Step 1: Apply the activation derivative.
            // delta was dE/d(activated_output). We need dE/d(pre_activation).
            // Since activated_output = activation(z), by the chain rule:
            //   dE/dz = dE/d(activated_output) * activation'(z)
            // This is an elementwise (Hadamard) product.
            let act_deriv = activate_deriv(&pre_activations[i], self.layers[i].activation);
            delta = &delta * &act_deriv;

            // Step 2: Propagate through the linear transform.
            // Since z = W * input + b, by the chain rule:
            //   dE/d(input) = W^T * dE/dz
            // This maps the gradient back from the layer's output space
            // to its input space.
            delta = self.layers[i].weight.t().dot(&delta);
        }

        // delta is now dE/dx -- the gradient of the energy w.r.t. the original input.
        delta
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
    fn test_gibbs_analytical_gradient_vs_finite_difference() {
        // SCENARIO-CORE-003: analytical gradient matches numerical
        for activation in [Activation::SiLU, Activation::ReLU, Activation::Tanh] {
            let model = GibbsModel::new(GibbsConfig {
                input_dim: 5,
                hidden_dims: vec![4, 3],
                activation,
                dropout: 0.0,
            })
            .unwrap();
            let x = Array1::random(5, Uniform::new(-0.5, 0.5));
            let grad = model.grad_energy(&x.view());

            // Compare with finite difference
            let eps: Float = 1e-4;
            for i in 0..5 {
                let mut x_p = x.clone();
                let mut x_m = x.clone();
                x_p[i] += eps;
                x_m[i] -= eps;
                let fd = (model.energy(&x_p.view()) - model.energy(&x_m.view())) / (2.0 * eps);
                assert!(
                    (grad[i] - fd).abs() < 0.05,
                    "Gradient mismatch for {:?} at index {i}: analytic={}, fd={fd}",
                    activation,
                    grad[i]
                );
            }
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
