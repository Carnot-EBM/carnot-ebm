"""ALPS (Annealed Langevin Posterior Sampling) sampler.

**Researcher summary:**
    Anneals static posterior distributions to overcome MCMC delays.
    Adds a temperature schedule to Langevin dynamics.
    
Spec: REQ-SAMPLE-2109
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp
import jax.random as jrandom

if TYPE_CHECKING:
    from carnot.phase3.continuous_ebm import ContinuousEBM


@dataclass
class AlpsSampler:
    """Annealed Langevin Posterior Sampling (ALPS).
    
    Provides faster convergence for ContinuousEBMs by scaling the noise schedule
    (temperature) down over time.
    
    Attributes:
        step_size: The discretization step size.
        clip_norm: Optional L2 gradient clipping.
        init_temp: Initial temperature for the noise schedule.
        final_temp: Final temperature for the noise schedule.
        
    Spec: REQ-SAMPLE-2109
    """
    step_size: float = 0.01
    clip_norm: float | None = None
    init_temp: float = 1.0
    final_temp: float = 0.01

    def _clip_gradient(self, grad: jax.Array) -> jax.Array:
        if self.clip_norm is None:
            return grad
        norm = jnp.linalg.norm(grad)
        return jnp.where(norm > self.clip_norm, grad * self.clip_norm / norm, grad)

    def sample(
        self,
        energy_fn: 'ContinuousEBM',
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
        cbf_fn: Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        if key is None:
            key = jrandom.PRNGKey(0)

        # Linear decay of temperature
        temps = jnp.linspace(self.init_temp, self.final_temp, n_steps)

        def step(
            carry: tuple[jax.Array, jax.Array], t_idx: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, subkey = jrandom.split(key)
            
            # For ContinuousEBM: E(x) = -0.5 * x^T J x - h^T x
            # grad E(x) = -J x - h
            grad = -jnp.dot(energy_fn.coupling, x) - energy_fn.bias
            
            if cbf_fn is not None:
                grad = grad + jax.grad(cbf_fn)(x)
                
            grad = self._clip_gradient(grad)
            
            temp = temps[t_idx]
            noise_scale = jnp.sqrt(self.step_size * temp)
            noise = jrandom.normal(subkey, x.shape)
            
            x_new = x - (self.step_size * 0.5) * grad + noise_scale * noise
            # Clip to [-1, 1] as typical for ContinuousEBM representing spins
            x_new = jnp.clip(x_new, -1.0, 1.0)
            return (x_new, key), x_new

        (x_final, _), _ = jax.lax.scan(step, (init, key), jnp.arange(n_steps), length=n_steps)
        return x_final

    def sample_chain(
        self,
        energy_fn: 'ContinuousEBM',
        init: jax.Array,
        n_steps: int,
        key: jax.Array | None = None,
        cbf_fn: Callable[[jax.Array], jax.Array] | None = None,
    ) -> jax.Array:
        if key is None:
            key = jrandom.PRNGKey(0)

        temps = jnp.linspace(self.init_temp, self.final_temp, n_steps)

        def step(
            carry: tuple[jax.Array, jax.Array], t_idx: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
            x, key = carry
            key, subkey = jrandom.split(key)
            
            grad = -jnp.dot(energy_fn.coupling, x) - energy_fn.bias
            
            if cbf_fn is not None:
                grad = grad + jax.grad(cbf_fn)(x)
                
            grad = self._clip_gradient(grad)
            
            temp = temps[t_idx]
            noise_scale = jnp.sqrt(self.step_size * temp)
            noise = jrandom.normal(subkey, x.shape)
            
            x_new = x - (self.step_size * 0.5) * grad + noise_scale * noise
            x_new = jnp.clip(x_new, -1.0, 1.0)
            return (x_new, key), x_new

        (_, _), chain = jax.lax.scan(step, (init, key), jnp.arange(n_steps), length=n_steps)
        return chain
