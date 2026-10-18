"""Denoising Thermodynamic Model (DTM) CPU Simulator.

Spec: REQ-SAMPLE-2067
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

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


class _IsingEnergyFunction:
    """Internal energy function adapter for Ising variables."""

    def __init__(self, biases: jax.Array, couplings: jax.Array):
        self.biases = biases
        self.couplings = couplings
        self._input_dim = biases.shape[0]

    def energy(self, x: jax.Array) -> jax.Array:
        return -jnp.dot(x, jnp.dot(self.couplings, x)) - jnp.dot(self.biases, x)

    def energy_batch(self, xs: jax.Array) -> jax.Array:
        return jax.vmap(self.energy)(xs)

    def grad_energy(self, x: jax.Array) -> jax.Array:
        return jax.grad(self.energy)(x)

    @property
    def input_dim(self) -> int:
        return self._input_dim


@dataclass
class DtmBackend:
    """DTM Simulator backend adapter satisfying SamplerBackend protocol."""

    seed: int = 42
    step_size: float = 0.01
    clip_norm: float | None = None

    @property
    def backend_name(self) -> str:
        return "dtm"

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        energy_fn = _IsingEnergyFunction(jnp.array(biases), jnp.array(couplings))
        sampler = DTMSampler(step_size=self.step_size, clip_norm=self.clip_norm)
        key = jrandom.PRNGKey(self.seed)

        # Schedule goes from 0.1 to beta
        beta_schedule = jnp.linspace(0.1, beta, n_steps)

        def sample_single(k: jax.Array) -> jax.Array:
            init_state = jrandom.uniform(k, shape=(biases.shape[0],))
            final_state = sampler.sample(energy_fn, init_state, n_steps, k, beta_schedule)
            return final_state > 0.5

        keys = jrandom.split(key, n_samples)
        samples = jax.vmap(sample_single)(keys)
        return np.asarray(samples)

    def set_constraints(self, constraints: Any) -> None:
        """No-op primal-dual hook for the DTM simulator backend.

        Spec: REQ-SAMPLE-2250
        """
        return None

    def dual_update_step(self, dual_lr: float) -> None:
        """No-op dual-update hook for the DTM simulator backend.

        Spec: REQ-SAMPLE-2250
        """
        return None

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        n_steps = int(config.get("n_steps", 100))
        energy_fn = _IsingEnergyFunction(jnp.array(biases), jnp.array(couplings))
        sampler = DTMSampler(step_size=self.step_size, clip_norm=self.clip_norm)
        key = jrandom.PRNGKey(self.seed)

        beta = float(config.get("beta", 1.0))
        beta_schedule = jnp.full(n_steps, beta)

        def sample_single(k: jax.Array) -> jax.Array:
            init_state = jrandom.uniform(k, shape=(biases.shape[0],))
            final_state = sampler.sample(energy_fn, init_state, n_steps, k, beta_schedule)
            return final_state > 0.5

        keys = jrandom.split(key, n_samples)
        samples = jax.vmap(sample_single)(keys)
        return np.asarray(samples)
