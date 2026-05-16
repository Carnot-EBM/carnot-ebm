"""Ising (small tier) Energy Based Model -- JAX implementation.

**Researcher summary:**
    Pairwise EBM with energy E(x) = -0.5 * x^T J x - b^T x. Coupling matrix J
    is symmetric. Smallest model tier in Carnot, suitable for low-dimensional
    problems and rapid prototyping.

**Detailed explanation for engineers:**
    The Ising model is the simplest Energy-Based Model in the Carnot hierarchy.
    It is inspired by the Ising model from statistical physics, where "spins"
    (binary variables) interact with their neighbors.

    **The energy function:**
        E(x) = -0.5 * x^T J x - b^T x

    Where:
    - x is the input configuration (a vector of real values)
    - J is the "coupling matrix" — a square, symmetric matrix that defines how
      pairs of variables interact. If J[i][j] is large and positive, variables
      x[i] and x[j] "want" to have the same sign (low energy when aligned).
    - b is the "bias vector" — if b[i] is large and positive, variable x[i]
      "wants" to be positive.
    - The -0.5 factor avoids double-counting since J is symmetric.

    **Why is it the "small" tier?**
    The Ising model has no hidden layers and no nonlinearities — it is a purely
    quadratic energy function. This means it can only capture pairwise
    correlations between variables. For richer distributions, use the Gibbs
    (medium) or Boltzmann (large) tiers. However, the Ising tier is fast,
    analytically tractable, and great for testing samplers and training methods.

    **Parameter count:** For input_dim=d, the coupling matrix has d*d entries
    (though only d*(d+1)/2 are independent due to symmetry) and the bias has d
    entries. For d=784 (MNIST), this is ~615K parameters.

Spec: REQ-TIER-001, REQ-TIER-004, REQ-TIER-005
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import AutoGradMixin


@dataclass
class IsingConfig:
    """Configuration for the Ising model.

    **Researcher summary:**
        Specifies visible dimension, optional hidden dimension (unused in base
        Ising), and coupling matrix initialization strategy.

    **Detailed explanation for engineers:**
        These are the architectural hyperparameters you choose before creating
        an Ising model. They cannot be changed after initialization.

    Attributes:
        input_dim: Number of visible units (dimensions of the configuration
            vector x). Default 784 corresponds to a flattened 28x28 image.
        hidden_dim: Reserved for restricted Boltzmann machine variants. Not
            used in the base Ising model. If specified, must be positive.
        coupling_init: How to initialize the coupling matrix J. Options:
            - "xavier_uniform": Random values scaled by sqrt(6 / (d + d)),
              which keeps the energy scale reasonable regardless of dimension.
              This is the default and recommended initialization.
            - "zeros": All-zero coupling matrix (no interactions). Useful for
              testing or when you plan to load trained parameters.

    For example::

        config = IsingConfig(input_dim=100, coupling_init="xavier_uniform")
        config.validate()  # raises ValueError if invalid

    Spec: REQ-TIER-005
    """

    input_dim: int = 784
    hidden_dim: int | None = None
    coupling_init: str = "xavier_uniform"

    def validate(self) -> None:
        """Validate configuration parameters.

        Raises ValueError if input_dim <= 0 or hidden_dim <= 0.

        Spec: SCENARIO-TIER-006
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0 if specified")


class IsingModel(AutoGradMixin):
    """Ising Energy Based Model with pairwise interactions.

    **Researcher summary:**
        Quadratic EBM: E(x) = -0.5 * x^T J x - b^T x. Inherits
        ``energy_batch`` and ``grad_energy`` from ``AutoGradMixin`` via
        ``jax.vmap`` and ``jax.grad`` respectively.

    **Detailed explanation for engineers:**
        This class represents a trained (or initialized) Ising model. It holds
        two sets of learnable parameters:

        1. ``self.coupling`` — A symmetric matrix J of shape (input_dim, input_dim).
           Encodes how each pair of variables interacts. Symmetry is enforced
           during initialization by averaging J with its transpose: (J + J^T) / 2.

        2. ``self.bias`` — A vector b of shape (input_dim,). Encodes the
           "preferred" value for each variable.

        Because this class inherits from ``AutoGradMixin``, you only see
        ``energy()`` defined here — ``grad_energy()`` and ``energy_batch()``
        are automatically provided by JAX's autodiff and vmap.

    For example::

        import jax.numpy as jnp

        model = IsingModel(IsingConfig(input_dim=10))
        x = jnp.ones(10)

        e = model.energy(x)        # scalar energy E(x)
        g = model.grad_energy(x)   # gradient dE/dx, shape (10,)
        batch_e = model.energy_batch(jnp.ones((5, 10)))  # shape (5,)

    Spec: REQ-TIER-001
    """

    def __init__(self, config: IsingConfig, key: jax.Array | None = None) -> None:
        """Create a new Ising model with initialized parameters.

        **Detailed explanation for engineers:**
            The constructor validates the config, then initializes the coupling
            matrix and bias vector. The PRNG key controls randomness — passing
            the same key produces the same initialization (reproducibility).

            **What is a PRNG key?**
            JAX uses explicit random number generator keys instead of global
            state (like numpy.random). You create a key with
            ``jax.random.PRNGKey(seed)`` and split it to get independent
            sub-keys. This makes randomness reproducible and compatible with
            JAX's functional programming model.

        Args:
            config: An IsingConfig specifying dimensions and initialization.
            key: JAX PRNG key for random initialization. If None, uses seed 0.

        Spec: REQ-TIER-001, SCENARIO-TIER-006
        """
        config.validate()
        self.config = config

        # Use a default PRNG key if none provided (seed=0 for reproducibility)
        if key is None:
            key = jrandom.PRNGKey(0)

        dim = config.input_dim
        # Split the key into two independent sub-keys: one for coupling, one reserved
        k1, k2 = jrandom.split(key)

        # Initialize coupling matrix (symmetric) based on the chosen strategy
        if config.coupling_init == "xavier_uniform":
            # Xavier/Glorot uniform initialization: scale by sqrt(6 / (fan_in + fan_out))
            # This keeps the variance of the energy roughly constant regardless of
            # the dimension, preventing exploding or vanishing energies.
            limit = jnp.sqrt(6.0 / (dim + dim))
            j = jrandom.uniform(k1, (dim, dim), minval=-limit, maxval=limit)
        elif config.coupling_init == "zeros":
            # All-zero initialization: no interactions. Model starts as a "blank slate".
            j = jnp.zeros((dim, dim))
        else:
            raise ValueError(f"Unknown initializer: {config.coupling_init}")

        # Enforce symmetry: J_ij = J_ji. This is required because the energy
        # formula x^T J x implicitly uses both J_ij and J_ji, so they must agree.
        self.coupling = (j + j.T) / 2.0
        # Bias initialized to zero — no preferred direction initially
        self.bias = jnp.zeros(dim)

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy E(x) = -0.5 * x^T J x - b^T x.

        **Researcher summary:**
            Quadratic energy: pairwise coupling term + linear bias term.

        **Detailed explanation for engineers:**
            The energy computation has two parts:

            1. ``x @ self.coupling @ x`` — This is x^T J x, a scalar that
               sums up all pairwise interactions: sum_ij J_ij * x_i * x_j.
               Multiplied by -0.5 so that aligned variables (same sign as
               their coupling) produce low (favorable) energy.

            2. ``self.bias @ x`` — This is b^T x, a scalar that sums up
               individual biases: sum_i b_i * x_i. Subtracted so that
               variables aligned with their bias produce low energy.

        Args:
            x: A 1-D JAX array of shape (input_dim,).

        Returns:
            A scalar JAX array representing the energy.

        Spec: REQ-CORE-002, SCENARIO-CORE-001, SCENARIO-TIER-001
        """
        # x @ self.coupling @ x computes x^T J x (scalar via two matrix-vector products)
        # self.bias @ x computes b^T x (dot product)
        return -0.5 * x @ self.coupling @ x - self.bias @ x

    @property
    def input_dim(self) -> int:
        """Number of visible units / input dimensions."""
        return self.config.input_dim

    def parameter_memory_bytes(self) -> int:
        """Compute total memory footprint of model parameters in bytes.

        **Detailed explanation for engineers:**
            Useful for checking that a model fits within memory budgets
            (especially relevant for the autoresearch evaluator's tertiary
            gate). Accounts for both the coupling matrix and bias vector,
            using their actual dtype's itemsize (4 bytes for float32,
            8 bytes for float64).

        Returns:
            Total bytes used by coupling matrix + bias vector.

        Spec: SCENARIO-TIER-005
        """
        itemsize = self.coupling.dtype.itemsize  # bytes per element (4 for f32)
        return self.coupling.size * itemsize + self.bias.size * itemsize
