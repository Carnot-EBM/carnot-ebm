"""Gibbs (medium tier) Energy Based Model -- JAX implementation.

**Researcher summary:**
    Multi-layer energy network: E(x) = w^T f_L(...f_2(f_1(x))) + b -> scalar.
    Supports SiLU, ReLU, and Tanh activations. Gradients computed automatically
    via JAX's reverse-mode autodiff (jax.grad).

**Detailed explanation for engineers:**
    The Gibbs model is the "middle child" of the Carnot hierarchy — more
    expressive than the Ising model (which is limited to pairwise quadratic
    interactions) but simpler than the Boltzmann model (which adds residual
    skip connections for very deep networks).

    **How it works at a high level:**
    Think of it as a standard neural network (multi-layer perceptron) that
    outputs a single number — the energy — instead of class probabilities.

    ```
    Input x (e.g., 784-dim image)
      → [Dense Layer 1] activation(W1 @ x + b1)
      → [Dense Layer 2] activation(W2 @ h1 + b2)
      → ... (as many layers as you configure) ...
      → [Output] w_out @ h_last + b_out  →  scalar energy E(x)
    ```

    Each "Dense Layer" does: output = activation(W * input + b).
    The final output layer has NO activation — it just linearly combines the
    last hidden layer into a single number.

    **What makes Gibbs different from Boltzmann?**
    - No residual (skip) connections — simpler architecture
    - Fewer layers typically (2-4 vs 4+)
    - Faster to evaluate but less expressive for very complex distributions
    - Good default choice for most practical problems

    **Activation functions:**
    - SiLU (default): x * sigmoid(x) — smooth, no dead zones
    - ReLU: max(0, x) — fast but can have dead neurons
    - Tanh: tanh(x) — bounded output, can have vanishing gradients

    **Why use AutoGradMixin instead of analytical backprop?**
    The Rust implementation uses hand-derived analytical backpropagation for
    maximum speed. In JAX, we get the same mathematical result for free via
    jax.grad — JAX traces the computation and automatically derives the
    gradient. This is exact (not numerical approximation) and nearly as fast
    as hand-coded backprop, with zero maintenance burden. The AutoGradMixin
    provides grad_energy via jax.grad and energy_batch via jax.vmap.

Spec: REQ-TIER-002, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


def _silu(x: jax.Array) -> jax.Array:
    """SiLU activation: x * sigmoid(x).

    Also known as "Swish". Smooth, differentiable everywhere, no dead zones.
    The default activation for Gibbs models.
    """
    return x * jax.nn.sigmoid(x)


def _apply_activation(x: jax.Array, activation: str) -> jax.Array:
    """Apply the named activation function elementwise.

    **For engineers:**
        This dispatches to the appropriate JAX activation based on the
        string name from the config. All three options are standard neural
        network activations — see the module docstring for when to use each.

    Args:
        x: Pre-activation values (output of W @ h + b).
        activation: One of "silu", "relu", "tanh".

    Returns:
        Activated values, same shape as x.
    """
    if activation == "silu":
        return _silu(x)
    elif activation == "relu":
        return jax.nn.relu(x)
    elif activation == "tanh":
        return jnp.tanh(x)
    else:
        raise ValueError(f"Unknown activation: {activation}. Use 'silu', 'relu', or 'tanh'.")


@dataclass
class GibbsConfig:
    """Configuration for the Gibbs model.

    **Researcher summary:**
        Specifies network topology (layer widths), activation function,
        and dropout rate. Default: 784 → [512, 256] → scalar with SiLU.

    **Detailed explanation for engineers:**
        These are the architectural hyperparameters you choose before creating
        a Gibbs model. They cannot be changed after initialization.

        - ``input_dim``: Size of the input vector. Default 784 corresponds
          to a flattened 28x28 image (like MNIST).

        - ``hidden_dims``: List of hidden layer sizes. For example,
          [512, 256] means "first hidden layer has 512 neurons, second has
          256." The network progressively compresses the representation.
          More/wider layers = more expressive but slower and more memory.

        - ``activation``: Which nonlinear function to apply after each
          hidden layer. Options: "silu" (default, recommended), "relu"
          (fast but can have dead neurons), "tanh" (bounded output).

        - ``dropout``: Fraction of neurons to randomly zero out during
          training (regularization). Must be in [0, 1). Set to 0.0 to
          disable. NOTE: Dropout is reserved for future training support
          and is not currently applied during inference.

    For example::

        config = GibbsConfig(input_dim=100, hidden_dims=[64, 32])
        config.validate()  # raises ValueError if invalid

    Spec: REQ-TIER-005
    """

    input_dim: int = 784
    hidden_dims: list[int] | None = None
    activation: str = "silu"
    dropout: float = 0.0

    def __post_init__(self) -> None:
        """Set default hidden_dims if not provided."""
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256]

    def validate(self) -> None:
        """Validate configuration parameters.

        **For engineers:**
            Checks that all dimensions are positive, hidden_dims is not empty,
            activation is recognized, and dropout is in valid range.

        Raises:
            ValueError: If any parameter is invalid.

        Spec: SCENARIO-TIER-006
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if not self.hidden_dims or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must have at least one layer")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("all hidden_dims must be > 0")
        if self.activation not in ("silu", "relu", "tanh"):
            raise ValueError(
                f"Unknown activation: {self.activation}. Use 'silu', 'relu', or 'tanh'."
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")


class GibbsModel(AutoGradMixin):
    """Gibbs (medium tier) Energy Based Model — multi-layer energy network.

    **Researcher summary:**
        MLP energy function: E(x) = w^T f_L(W_L f_{L-1}(...) + b_L) + b_out.
        Gradients via jax.grad (automatic reverse-mode AD). Batch via jax.vmap.

    **Detailed explanation for engineers:**
        This model stacks dense layers to create a nonlinear energy function.
        Unlike the Ising model (which is purely quadratic), the Gibbs model
        can learn complex, multi-modal energy landscapes.

        The forward pass is:
        1. Pass input through each hidden layer: h = activation(W @ h + b)
        2. Linear readout to scalar: E = w_out @ h_last + b_out

        Parameters stored:
        - ``self.layers``: List of (weight, bias) tuples for hidden layers
        - ``self.output_weight``: Final weight vector for scalar readout
        - ``self.output_bias``: Scalar bias for the output

        Because this class inherits from ``AutoGradMixin``, you only need
        to implement ``energy(x)`` — ``grad_energy(x)`` and
        ``energy_batch(xs)`` are automatically provided by JAX.

    For example::

        model = GibbsModel(GibbsConfig(input_dim=10, hidden_dims=[8, 4]))
        x = jnp.ones(10)
        energy = model.energy(x)       # scalar
        grad = model.grad_energy(x)    # 10-dim gradient
        batch = model.energy_batch(jnp.ones((5, 10)))  # 5 energies

    Spec: REQ-TIER-002
    """

    def __init__(
        self,
        config: GibbsConfig,
        key: jax.Array | None = None,
    ) -> None:
        """Create a new Gibbs model with initialized parameters.

        **Detailed explanation for engineers:**
            The constructor validates the config, then builds the network
            layer by layer. For each pair of consecutive dimensions
            (input_dim → hidden_dims[0], hidden_dims[0] → hidden_dims[1],
            etc.), it creates a weight matrix and bias vector.

            **Initialization strategy (Xavier/Glorot uniform):**
            Weight values are drawn uniformly from [-limit, +limit] where
            limit = sqrt(6 / (fan_in + fan_out)). This keeps the variance
            of activations roughly constant across layers, preventing
            exploding or vanishing values in deep networks.

            **PRNG key (JAX randomness):**
            JAX uses explicit random number generator keys instead of
            global state. You create a key with jax.random.PRNGKey(seed)
            and split it to get independent sub-keys. Passing the same
            key produces the same initialization (reproducibility).

        Args:
            config: A GibbsConfig specifying dimensions and activation.
            key: JAX PRNG key for random initialization. If None, uses seed 0.

        Raises:
            ValueError: If config has invalid values.

        Spec: REQ-TIER-002, REQ-TIER-005, SCENARIO-TIER-006
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        hidden_dims = config.hidden_dims
        assert hidden_dims is not None  # validated above

        # Build hidden layers: each transforms prev_dim → hidden_dim
        self.layers: list[tuple[jax.Array, jax.Array]] = []
        prev_dim = config.input_dim

        for hidden_dim in hidden_dims:
            k_w, key = jrandom.split(key)
            # Xavier uniform initialization: scale by sqrt(6 / (fan_in + fan_out))
            limit = jnp.sqrt(6.0 / (prev_dim + hidden_dim))
            weight = jrandom.uniform(
                k_w, (hidden_dim, prev_dim), minval=-limit, maxval=limit
            )
            bias = jnp.zeros(hidden_dim)
            self.layers.append((weight, bias))
            prev_dim = hidden_dim

        # Output layer: dot product from last hidden dim → scalar energy.
        # Zero-initialized so the model starts with energy = 0 for all inputs.
        self.output_weight = jnp.zeros(prev_dim)
        self.output_bias = 0.0

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy E(x) for a single input.

        **Researcher summary:**
            Forward pass: x → dense layers with activation → linear → scalar.

        **Detailed explanation for engineers:**
            The energy is computed by passing the input through all hidden
            layers in sequence, then taking a dot product with the output
            weight vector:

            1. For each hidden layer: h = activation(W @ h + b)
               - W @ h is a matrix-vector multiply (linear transform)
               - + b shifts the result (bias)
               - activation adds nonlinearity (SiLU, ReLU, or Tanh)

            2. E = w_out @ h_last + b_out
               - Collapses the final hidden vector to a single scalar

            The result is a single number: the energy of the input
            configuration. Low energy = "likely/natural", high energy =
            "unlikely/unnatural".

        Args:
            x: A 1-D JAX array of shape (input_dim,).

        Returns:
            A scalar JAX array representing the energy.

        Spec: REQ-CORE-002, SCENARIO-CORE-001, SCENARIO-TIER-002
        """
        h = x
        for weight, bias in self.layers:
            h = _apply_activation(weight @ h + bias, self.config.activation)

        return self.output_weight @ h + self.output_bias

    @property
    def input_dim(self) -> int:
        """Number of input dimensions."""
        return self.config.input_dim

    def parameter_memory_bytes(self) -> int:
        """Compute total memory footprint of model parameters in bytes.

        **For engineers:**
            Sums up the memory used by all weight matrices, bias vectors,
            and the output layer. Uses the actual dtype's itemsize (4 bytes
            for float32, 8 for float64). Useful for the autoresearch
            evaluator's tertiary gate (memory budget check).

        Returns:
            Total bytes used by all learnable parameters.

        Spec: SCENARIO-TIER-005
        """
        total = 0
        for weight, bias in self.layers:
            itemsize = weight.dtype.itemsize
            total += weight.size * itemsize + bias.size * itemsize
        # Output weight + bias (bias is a Python float, count as one element)
        if hasattr(self.output_weight, "dtype"):
            itemsize = self.output_weight.dtype.itemsize
            total += self.output_weight.size * itemsize + itemsize
        return total
