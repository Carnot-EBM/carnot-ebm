"""NablaETS sampler for deep energy-guided decoding.

Spec: REQ-VERIFY-1690, SCENARIO-VERIFY-1690
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from carnot.samplers.langevin import LangevinSampler

if TYPE_CHECKING:
    from carnot.core.energy import EnergyFunction


@dataclass
class NablaETSConfig:
    """Configuration for NablaETS sampler.

    Attributes:
        K_steps: Number of steps for continuous Langevin dynamics. This is the 
            scaling parameter for Test-Time Scaling (ETS).
        step_size: Step size for the Langevin dynamics.
        clip_norm: Optional maximum gradient norm for clipping.
    """
    K_steps: int = 50
    step_size: float = 0.01
    clip_norm: float | None = None


class NablaETS:
    """NablaETS sampler that optimizes latent state guided by EBCN energy before token decoding.

    This combines the Nabla-Reasoner continuous optimization with Test-Time Scaling (ETS).
    At each token decoding step, instead of searching over discrete tokens, it applies
    Langevin dynamics in the continuous logit (or latent) space to minimize the
    EBCN energy function.

    Spec: REQ-VERIFY-1690
    """

    def __init__(self, config: NablaETSConfig | None = None) -> None:
        self.config = config if config is not None else NablaETSConfig()
        self._langevin = LangevinSampler(
            step_size=self.config.step_size,
            clip_norm=self.config.clip_norm
        )

    def optimize_latent_state(
        self,
        energy_fn: EnergyFunction,
        init_latent: jax.Array,
        key: jax.Array | None = None
    ) -> jax.Array:
        """Optimize the latent state using continuous Langevin dynamics.

        Args:
            energy_fn: Differentiable EBCN energy function over continuous states.
            init_latent: Initial latent state.
            key: JAX PRNG key.

        Returns:
            The optimized latent state after `K_steps` of Langevin dynamics.

        Spec: REQ-VERIFY-1690, SCENARIO-VERIFY-1690
        """
        return self._langevin.sample(
            energy_fn=energy_fn,
            init=init_latent,
            n_steps=self.config.K_steps,
            key=key
        )
