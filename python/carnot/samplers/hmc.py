"""Hamiltonian Monte Carlo sampler — JAX implementation.

Spec: REQ-SAMPLE-002
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom

from carnot.core.energy import EnergyFunction


@dataclass
class HMCSampler:
    """Hamiltonian Monte Carlo sampler.

    Spec: REQ-SAMPLE-002
    """

    step_size: float = 0.1
    num_leapfrog_steps: int = 10

    def _leapfrog(
        self,
        energy_fn: EnergyFunction,
        x: jax.Array,
        p: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Leapfrog integration."""
        grad = energy_fn.grad_energy(x)
        p = p - (self.step_size * 0.5) * grad

        def body(i: int, carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            x, p = carry
            x = x + self.step_size * p
            grad = energy_fn.grad_energy(x)
            # Full momentum step except at last iteration
            p = p - self.step_size * grad * jnp.where(i < self.num_leapfrog_steps - 1, 1.0, 0.5)
            return x, p

        x_new, p_new = jax.lax.fori_loop(0, self.num_leapfrog_steps, body, (x, p))
        return x_new, p_new

    def sample(
        self,
        energy_fn: EnergyFunction,
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Run HMC, return final sample.

        Spec: REQ-SAMPLE-002, SCENARIO-SAMPLE-002
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        def step(carry: tuple[jax.Array, jax.Array], _: None) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, k1, k2 = jrandom.split(key, 3)
            p = jrandom.normal(k1, x.shape)
            h_old = energy_fn.energy(x) + 0.5 * jnp.sum(p**2)
            x_new, p_new = self._leapfrog(energy_fn, x, p)
            h_new = energy_fn.energy(x_new) + 0.5 * jnp.sum(p_new**2)
            accept_prob = jnp.exp(jnp.minimum(-h_new + h_old, 0.0))
            u = jrandom.uniform(k2)
            x_out = jnp.where(u < accept_prob, x_new, x)
            return (x_out, key), x_out

        (x_final, _), _ = jax.lax.scan(step, (init, key), None, length=n_steps)
        return x_final
