"""Ising (small tier) Energy Based Model — JAX implementation.

E(x) = -0.5 * x^T J x - b^T x

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

    Spec: REQ-TIER-005
    """

    input_dim: int = 784
    hidden_dim: int | None = None
    coupling_init: str = "xavier_uniform"

    def validate(self) -> None:
        """Validate configuration.

        Spec: SCENARIO-TIER-006
        """
        if self.input_dim <= 0:
            raise ValueError("input_dim must be > 0")
        if self.hidden_dim is not None and self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0 if specified")


class IsingModel(AutoGradMixin):
    """Ising Energy Based Model.

    E(x) = -0.5 * x^T J x - b^T x

    Spec: REQ-TIER-001
    """

    def __init__(self, config: IsingConfig, key: jax.Array | None = None) -> None:
        """Create a new Ising model.

        Spec: REQ-TIER-001, SCENARIO-TIER-006
        """
        config.validate()
        self.config = config

        if key is None:
            key = jrandom.PRNGKey(0)

        dim = config.input_dim
        k1, k2 = jrandom.split(key)

        # Initialize coupling matrix (symmetric)
        if config.coupling_init == "xavier_uniform":
            limit = jnp.sqrt(6.0 / (dim + dim))
            j = jrandom.uniform(k1, (dim, dim), minval=-limit, maxval=limit)
        elif config.coupling_init == "zeros":
            j = jnp.zeros((dim, dim))
        else:
            raise ValueError(f"Unknown initializer: {config.coupling_init}")

        # Make symmetric
        self.coupling = (j + j.T) / 2.0
        self.bias = jnp.zeros(dim)

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy.

        Spec: REQ-CORE-002, SCENARIO-CORE-001, SCENARIO-TIER-001
        """
        return -0.5 * x @ self.coupling @ x - self.bias @ x

    @property
    def input_dim(self) -> int:
        return self.config.input_dim

    def parameter_memory_bytes(self) -> int:
        """Memory footprint in bytes.

        Spec: SCENARIO-TIER-005
        """
        itemsize = self.coupling.dtype.itemsize
        return self.coupling.size * itemsize + self.bias.size * itemsize
