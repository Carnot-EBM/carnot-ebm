"""Flow Sampling Density Estimator Prototype.

**Researcher summary:**
    Implements a conditional denoising process for unnormalized target
    densities (e.g., EBMs). Uses a simple probability flow ODE or
    Langevin-like reverse step informed by the exact score
    (grad_x log p(x) = -grad_E(x)).

Spec: REQ-SAMPLE-1960
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
class FlowSampler:
    """Flow Sampling prototype for unnormalized densities.

    Implements a basic forward/reverse conditional denoising process.
    
    Spec: REQ-SAMPLE-1960
    """

    n_steps: int = 100
    dt: float = 0.01

    def forward_step(
        self, x0: jax.Array, t: float, key: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Conditional denoising forward step.

        Adds Gaussian noise scaled by sqrt(t) to simulate the diffusion process.
        """
        noise = jrandom.normal(key, x0.shape)
        x_t = x0 + jnp.sqrt(t) * noise
        return x_t, noise

    def sample(
        self,
        energy_fn: EnergyFunction,
        shape: tuple[int, ...],
        key: jax.Array | None = None,
    ) -> jax.Array:
        """Reverse step sampling from noise to target density.

        Uses an Euler-Maruyama discretization of the reverse SDE informed
        by the target unnormalized density score.
        """
        if key is None:
            key = jrandom.PRNGKey(0)

        # Start from prior N(0, I)
        key, subkey = jrandom.split(key)
        x = jrandom.normal(subkey, shape)

        def step_fn(
            carry: tuple[jax.Array, jax.Array], _: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            x_curr, k = carry
            k, sk = jrandom.split(k)
            # The score of the target is -grad_E(x)
            score = -energy_fn.grad_energy(x_curr)
            # Reverse ODE/SDE step
            x_next = x_curr - 0.5 * score * self.dt
            noise = jrandom.normal(sk, x_curr.shape)
            x_next = x_next + jnp.sqrt(self.dt) * noise
            return (x_next, k), None

        (x_final, _), _ = jax.lax.scan(
            step_fn,
            (x, key),
            jnp.arange(self.n_steps),
        )
        return x_final
