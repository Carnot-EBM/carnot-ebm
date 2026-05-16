"""Denoising Thermodynamic Model (DTM) CPU Simulator.

Spec: REQ-SAMPLE-2067
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import jax.random as jrandom

if TYPE_CHECKING:
    from carnot.core.energy import EnergyFunction


@dataclass
class DTMSampler:
    """DTM Simulator integrating Langevin thermal noise.
    
    Implements a thermal noise profile mapping to state updates (arXiv:2510.23972).
    """

    step_size: float = 0.01
    clip_norm: float | None = None

    def _clip_gradient(self, grad: jax.Array) -> jax.Array:
        if self.clip_norm is None:
            return grad
        norm = jnp.linalg.norm(grad)
        return jnp.where(norm > self.clip_norm, grad * self.clip_norm / norm, grad)

    def sample_chain(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
        beta_schedule: jax.Array | None = None,
    ) -> jax.Array:
        if key is None:
            key = jrandom.PRNGKey(0)

        if beta_schedule is None:
            # Default thermal noise profile: linear annealing from beta=0.1 to 1.0
            beta_schedule = jnp.linspace(0.1, 1.0, n_steps)

        def step(
            carry: tuple[jax.Array, jax.Array], i: int
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, subkey = jrandom.split(key)
            
            beta = beta_schedule[i]
            
            grad = energy_fn.grad_energy(x)
            grad = self._clip_gradient(grad)
            
            noise_scale = jnp.sqrt(self.step_size / beta)
            force_scale = self.step_size / 2.0
            
            noise = jrandom.normal(subkey, x.shape)
            x_new = x - force_scale * grad + noise_scale * noise
            
            return (x_new, key), x_new

        steps_idx = jnp.arange(n_steps)
        (_, _), chain = jax.lax.scan(step, (init, key), steps_idx, length=n_steps)
        return chain

    def sample(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
        beta_schedule: jax.Array | None = None,
    ) -> jax.Array:
        chain = self.sample_chain(energy_fn, init, n_steps, key, beta_schedule)
        return chain[-1]
