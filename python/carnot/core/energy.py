"""Energy function protocol and mixins.

Spec: REQ-CORE-002
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class EnergyFunction(Protocol):
    """Protocol for energy-based models.

    All model tiers implement this protocol.
    Spec: REQ-CORE-002
    """

    def energy(self, x: jax.Array) -> jax.Array:
        """Compute scalar energy for input x.

        Spec: REQ-CORE-002, SCENARIO-CORE-001
        """
        ...

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Compute energy for batch of inputs.

        Spec: REQ-CORE-002, SCENARIO-CORE-002
        """
        ...

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy w.r.t. x.

        Spec: REQ-CORE-002, SCENARIO-CORE-003
        """
        ...

    @property
    def input_dim(self) -> int:
        """Number of input dimensions."""
        ...


class AutoGradMixin:
    """Mixin that auto-derives grad_energy from energy using jax.grad.

    Spec: REQ-CORE-002
    """

    def energy(self, x: jax.Array) -> jax.Array:
        raise NotImplementedError

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        """Default batched energy via vmap.

        Spec: SCENARIO-CORE-002
        """
        return jax.vmap(self.energy)(xs)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Auto-derived gradient via jax.grad.

        Spec: SCENARIO-CORE-003
        """
        return jax.grad(self.energy)(x)
