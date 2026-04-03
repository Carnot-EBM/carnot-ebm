"""Langevin Dynamics sampler — JAX implementation.

x_{t+1} = x_t - (step_size/2) * grad_energy(x_t) + sqrt(step_size) * noise

Spec: REQ-SAMPLE-001
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import EnergyFunction


@dataclass
class LangevinSampler:
    """Unadjusted Langevin Dynamics sampler.

    Spec: REQ-SAMPLE-001
    """

    step_size: float = 0.01

    def sample(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run Langevin dynamics, return final sample.

        Spec: REQ-SAMPLE-001, SCENARIO-SAMPLE-001
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        noise_scale = jnp.sqrt(self.step_size)

        def step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, subkey = jrandom.split(key)
            grad = energy_fn.grad_energy(x)
            noise = jrandom.normal(subkey, x.shape)
            x_new = x - (self.step_size * 0.5) * grad + noise_scale * noise
            return (x_new, key), x_new

        (x_final, _), chain = jax.lax.scan(step, (init, key), None, length=n_steps)
        return x_final

    def sample_chain(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run Langevin dynamics, return full chain.

        Returns array of shape (n_steps, *init.shape).

        Spec: REQ-SAMPLE-001, SCENARIO-SAMPLE-001
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        noise_scale = jnp.sqrt(self.step_size)

        def step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, subkey = jrandom.split(key)
            grad = energy_fn.grad_energy(x)
            noise = jrandom.normal(subkey, x.shape)
            x_new = x - (self.step_size * 0.5) * grad + noise_scale * noise
            return (x_new, key), x_new

        (_, _), chain = jax.lax.scan(step, (init, key), None, length=n_steps)
        return chain
