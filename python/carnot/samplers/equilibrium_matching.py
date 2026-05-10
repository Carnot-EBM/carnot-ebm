"""Equilibrium Matching sampler -- JAX implementation.

Equilibrium Matching (EqM) is an optimization-oriented sampler for energy
landscapes where the goal is to find a low-energy constraint-satisfying state,
not to draw unbiased Boltzmann samples. Unlike Langevin dynamics, EqM does not
inject diffusion noise. It learns a smoothed equilibrium gradient online and
uses that learned field to move steadily toward states where the energy
gradient vanishes.

Spec: REQ-SAMPLE-1727
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp


@dataclass
class EquilibriumMatchingSampler:
    """Online equilibrium-gradient matcher for continuous EnergyFunctions.

    The sampler keeps a learned gradient vector with the same shape as the
    current state. Each step updates that vector as an exponential moving
    average of the current clipped energy gradient, blends the learned and
    instantaneous gradients, and moves against the blended direction. Momentum
    is optional and defaults to zero for predictable constraint satisfaction.

    Attributes:
        step_size: Size of each descent step.
        learning_rate: EMA update rate for the learned equilibrium gradient.
        matching_strength: Blend weight for the learned gradient. A value of
            0 uses the instantaneous gradient only; 1 uses the learned gradient.
        momentum: Heavy-ball velocity carryover for repeated EqM steps.
        clip_norm: Optional maximum L2 norm for raw energy gradients.

    Spec: REQ-SAMPLE-1727
    """

    step_size: float = 0.1
    learning_rate: float = 0.5
    matching_strength: float = 0.5
    momentum: float = 0.0
    clip_norm: float | None = None

    def _clip_gradient(self, grad: jax.Array) -> jax.Array:
        """Bound gradient norm while preserving direction.

        Spec: REQ-SAMPLE-1727-4
        """
        if self.clip_norm is None:
            return grad
        norm = jnp.linalg.norm(grad)
        safe_norm = jnp.maximum(norm, jnp.asarray(1e-12, dtype=grad.dtype))
        clipped = grad * (self.clip_norm / safe_norm)
        return jnp.where(norm > self.clip_norm, clipped, grad)

    def _update_learned_gradient(self, learned_gradient: jax.Array, grad: jax.Array) -> jax.Array:
        """EMA update for the learned equilibrium gradient.

        Spec: REQ-SAMPLE-1727-2
        """
        return (1.0 - self.learning_rate) * learned_gradient + self.learning_rate * grad

    def _matched_gradient(self, grad: jax.Array, learned_gradient: jax.Array) -> jax.Array:
        """Blend instantaneous and learned gradients into the EqM direction.

        Spec: REQ-SAMPLE-1727-2
        """
        return (1.0 - self.matching_strength) * grad + self.matching_strength * learned_gradient

    def _run(
        self,
        energy_fn: Any,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None,
    ) -> tuple[jax.Array, jax.Array]:
        """Run EqM and return both the final state and full chain."""
        del key  # EqM is deterministic; key is accepted for sampler interface parity.
        init = jnp.asarray(init)
        zeros = jnp.zeros_like(init)

        def step(
            carry: tuple[jax.Array, jax.Array, jax.Array], _: None
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
            x, learned_gradient, velocity = carry
            grad = self._clip_gradient(energy_fn.grad_energy(x))
            learned_gradient = self._update_learned_gradient(learned_gradient, grad)
            matched_gradient = self._matched_gradient(grad, learned_gradient)
            velocity = self.momentum * velocity - self.step_size * matched_gradient
            x_new = x + velocity
            return (x_new, learned_gradient, velocity), x_new

        (x_final, _, _), chain = jax.lax.scan(step, (init, zeros, zeros), None, length=n_steps)
        return x_final, chain

    def sample(
        self,
        energy_fn: Any,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run Equilibrium Matching and return only the final state.

        Spec: REQ-SAMPLE-1727-3, SCENARIO-SAMPLE-1727
        """
        x_final, _ = self._run(energy_fn, init, n_steps, key)
        return x_final

    def sample_chain(
        self,
        energy_fn: Any,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run Equilibrium Matching and return all intermediate states.

        Spec: REQ-SAMPLE-1727-3, SCENARIO-SAMPLE-1727
        """
        _, chain = self._run(energy_fn, init, n_steps, key)
        return chain
