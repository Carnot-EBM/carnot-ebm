"""Boltzmann (large tier) Energy Based Model -- JAX implementation.

**Researcher summary:**
    Deep residual energy network with SiLU activations. Architecture:
    input projection → [residual block]* → linear output (scalar energy).
    Each residual block: y = SiLU(W2 @ SiLU(W1 @ x + b1) + b2) + skip(x).
    Skip path uses a projection matrix when input/output dims differ.
    Analytical gradients via jax.grad auto-differentiation.

**Detailed explanation for engineers:**
    The Boltzmann model is the most powerful tier in Carnot. It uses a deep
    neural network with residual (skip) connections to define the energy
    landscape, similar to how ResNet uses residual connections for image
    classification — except the output is a single scalar (energy) instead
    of class probabilities.

    **Why residual connections matter:**
    In a plain deep network, gradients can "vanish" (shrink to near-zero)
    as they pass through many layers during backpropagation. This makes
    deep networks hard to train. Residual connections solve this by adding
    a "shortcut" that lets the gradient flow directly from output to input:

        output = main_path(x) + x    (residual connection)

    The gradient through the shortcut is always 1 (the identity), so it
    never vanishes. The network only needs to learn the "residual" — the
    difference between the input and the desired output.

    **Architecture:**
    ```
    Input x (e.g., 784-dim)
      → Input Projection (linear + SiLU) → hidden_dims[0]
      → Residual Block 1: hidden_dims[0] → hidden_dims[1]
      → Residual Block 2: hidden_dims[1] → hidden_dims[2]
      → ...
      → Linear output → scalar energy
    ```

    Each residual block has two paths:
    - Main: SiLU(W2 @ SiLU(W1 @ x + b1) + b2)
    - Skip: x (or projection(x) if dimensions change)
    - Output: main + skip

    **When to use Boltzmann vs Gibbs vs Ising:**
    - Ising: < 1000 dims, pairwise interactions only, fast
    - Gibbs: moderate complexity, 2-4 layers, good default
    - Boltzmann: high complexity, 4+ layers, residual connections prevent
      gradient vanishing, use when Gibbs underfits

Spec: REQ-TIER-003, REQ-TIER-004, REQ-TIER-005, REQ-TIER-006
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


def _silu(x: jax.Array) -> jax.Array:
    """SiLU activation: x * sigmoid(x). Smooth, non-monotonic, unbounded."""
    return x * jax.nn.sigmoid(x)


@dataclass
class BoltzmannConfig:
    """Configuration for the Boltzmann model.

    **Researcher summary:**
        Deep residual architecture config: input_dim, hidden layer widths,
        attention heads (reserved), residual toggle.

    **Detailed explanation for engineers:**
        - ``input_dim``: Size of the input vector.
        - ``hidden_dims``: List of hidden layer sizes. Each adjacent pair
          creates one residual block. E.g., [1024, 512, 256] → 2 blocks.
        - ``num_heads``: Reserved for future multi-head attention.
        - ``residual``: If True, use skip connections. If False, plain deep net.

    Spec: REQ-TIER-005
    """

    input_dim: int = 784
    hidden_dims: list[int] | None = None
    num_heads: int = 4
    residual: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dims is None:
            self.hidden_dims = [1024, 512, 256, 128]

    def validate(self) -> None:
        """Validate configuration.

        Spec: SCENARIO-TIER-006
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if not self.hidden_dims or len(self.hidden_dims) == 0:
            raise ValueError("hidden_dims must have at least one layer")
        if any(d <= 0 for d in self.hidden_dims):
            raise ValueError("all hidden_dims must be > 0")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be > 0")


class _ResidualBlock:
    """A single residual block: output = SiLU(W2 @ SiLU(W1 @ x + b1) + b2) + skip(x).

    **For engineers:**
        Two paths through the block:
        - Main path: two linear transforms with SiLU activations
        - Skip path: identity (same dims) or linear projection (different dims)
        Output is the sum of both paths.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_residual: bool,
        key: jax.Array,
    ) -> None:
        k1, k2, k3 = jrandom.split(key, 3)
        # Xavier initialization for both weight matrices
        limit1 = jnp.sqrt(6.0 / (in_dim + in_dim))
        limit2 = jnp.sqrt(6.0 / (in_dim + out_dim))

        # Main path weights
        self.w1 = jrandom.uniform(k1, (in_dim, in_dim), minval=-limit1, maxval=limit1)
        self.b1 = jnp.zeros(in_dim)
        self.w2 = jrandom.uniform(k2, (out_dim, in_dim), minval=-limit2, maxval=limit2)
        self.b2 = jnp.zeros(out_dim)

        # Skip path projection (only if dimensions differ)
        self.use_residual = use_residual
        if use_residual and in_dim != out_dim:
            limit_proj = jnp.sqrt(6.0 / (in_dim + out_dim))
            self.proj = jrandom.uniform(k3, (out_dim, in_dim), minval=-limit_proj, maxval=limit_proj)
        else:
            self.proj = None

    def forward(self, x: jax.Array) -> jax.Array:
        """Forward pass through the residual block.

        Main path: SiLU(W2 @ SiLU(W1 @ x + b1) + b2)
        Skip path: x (or proj @ x if dimensions differ)
        Output: main + skip
        """
        # Main path: two layers with SiLU
        h = _silu(self.w1 @ x + self.b1)
        out = _silu(self.w2 @ h + self.b2)

        # Skip path: add input (with optional projection)
        if self.use_residual:
            skip = self.proj @ x if self.proj is not None else x
            return out + skip
        return out


class BoltzmannModel(AutoGradMixin):
    """Boltzmann (large tier) Energy Based Model — deep residual energy network.

    **Researcher summary:**
        Deep residual EBM. Forward: input_proj + SiLU → [ResBlock]* → linear → scalar.
        Gradients via jax.grad auto-differentiation.

    **Detailed explanation for engineers:**
        This model stacks residual blocks to create a deep energy function.
        The depth allows it to learn complex, multi-modal energy landscapes
        that shallower models (Ising, Gibbs) cannot represent.

        The forward pass is:
        1. Project input to first hidden dimension + SiLU activation
        2. Pass through each residual block (each has two linear layers + skip)
        3. Linear readout to scalar energy

        Gradients are computed automatically by JAX's auto-differentiation,
        which is exact (not numerical approximation) and efficient (reverse-mode AD).

    For example::

        model = BoltzmannModel(BoltzmannConfig(input_dim=10, hidden_dims=[8, 6, 4]))
        x = jnp.ones(10)
        energy = model.energy(x)       # scalar
        grad = model.grad_energy(x)    # 10-dim gradient

    Spec: REQ-TIER-003
    """

    def __init__(
        self,
        config: BoltzmannConfig,
        key: jax.Array | None = None,
    ) -> None:
        """Create a Boltzmann model.

        Args:
            config: Model architecture configuration.
            key: JAX PRNG key for weight initialization. Uses default if None.

        Raises:
            ValueError: If config has invalid values (e.g., input_dim=0).

        Spec: REQ-TIER-003, SCENARIO-TIER-006
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        hidden_dims = config.hidden_dims
        assert hidden_dims is not None  # validated above

        # Input projection: input_dim → hidden_dims[0]
        k_proj, key = jrandom.split(key)
        limit = jnp.sqrt(6.0 / (config.input_dim + hidden_dims[0]))
        self.input_proj = jrandom.uniform(
            k_proj, (hidden_dims[0], config.input_dim),
            minval=-limit, maxval=limit,
        )
        self.input_bias = jnp.zeros(hidden_dims[0])

        # Residual blocks: one per adjacent pair of hidden dims
        self.blocks: list[_ResidualBlock] = []
        for i in range(len(hidden_dims) - 1):
            k_block, key = jrandom.split(key)
            block = _ResidualBlock(
                in_dim=hidden_dims[i],
                out_dim=hidden_dims[i + 1],
                use_residual=config.residual,
                key=k_block,
            )
            self.blocks.append(block)

        # Output: dot product with weight vector → scalar energy
        self.output_weight = jnp.zeros(hidden_dims[-1])
        self.output_bias = 0.0

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy for input x.

        Forward pass: input_proj + SiLU → blocks → linear → scalar.

        Spec: REQ-CORE-002, SCENARIO-TIER-003
        """
        # Input projection + SiLU activation
        h = _silu(self.input_proj @ x + self.input_bias)

        # Pass through residual blocks
        for block in self.blocks:
            h = block.forward(h)

        # Linear readout to scalar
        return self.output_weight @ h + self.output_bias

    @property
    def input_dim(self) -> int:
        return self.config.input_dim
