"""Denoising Thermodynamic Model (DTM) Backend.

**Researcher summary:**
    Simulates a continuous, denoising thermodynamic process for sample generation.
    Repurposes EBMs as diffusion-like steps.

Spec: REQ-SAMPLE-038
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from carnot.samplers.backend import SamplerBackend

logger = logging.getLogger(__name__)


@dataclass
class DtmBackend:
    """Denoising Thermodynamic Model backend for continuous sampling trajectories.
    
    Spec: REQ-SAMPLE-038
    """

    seed: int = 42
    _key: jax.Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._key = jrandom.PRNGKey(self.seed)

    @property
    def backend_name(self) -> str:
        return "dtm"

    def _next_key(self) -> jax.Array:
        self._key, subkey = jrandom.split(self._key)
        return subkey

    def minimize_energy(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        n_steps: int,
        beta: float,
    ) -> np.ndarray:
        """Run denoising trajectory to find low-energy states."""
        return self.sample(
            biases, 
            couplings, 
            n_samples, 
            {"beta": beta, "steps": n_steps, "anneal": True}
        )

    def sample(
        self,
        biases: np.ndarray,
        couplings: np.ndarray,
        n_samples: int,
        config: dict[str, Any],
    ) -> np.ndarray:
        """Draw samples using sequential denoising trajectory."""
        beta = float(config.get("beta", 1.0))
        steps = int(config.get("steps", 100))
        anneal = bool(config.get("anneal", False))
        
        b = jnp.asarray(biases, dtype=jnp.float32)
        J = jnp.asarray(couplings, dtype=jnp.float32)
        n_spins = biases.shape[0]

        @jax.jit
        def step_fn(x: jax.Array, key: jax.Array, current_beta: float) -> jax.Array:
            # Gradient of continuous energy landscape
            # E = -0.5 * x^T J x - b^T x
            grad_E = - jnp.dot(x, J) - b
            
            # Simple continuous relaxation step (Langevin-like)
            dt = 0.01
            noise = jrandom.normal(key, shape=x.shape)
            dx = - grad_E * dt + jnp.sqrt(2 * dt / current_beta) * noise
            
            x_new = jnp.clip(x + dx, 0.0, 1.0)
            return x_new

        # Initial random continuous state
        x = jrandom.uniform(self._next_key(), shape=(n_samples, n_spins), minval=0.0, maxval=1.0)
        
        # Denoising schedule
        betas = jnp.linspace(0.1, beta, steps) if anneal else jnp.full(steps, beta)
        
        for i in range(steps):
            x = step_fn(x, self._next_key(), float(betas[i]))
            
        # Final threshold to produce boolean samples as required by SamplerBackend protocol
        samples = (x > 0.5).astype(bool)
        return np.asarray(samples)

