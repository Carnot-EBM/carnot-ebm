"""Denoising Score Matching (DSM) training for Energy Based Models.

Spec: REQ-TRAIN-002

DSM loss:
  L = E_x E_noise [ || grad_energy(x + noise) - (-noise / sigma^2) ||^2 ]
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp


@runtime_checkable
class HasGradEnergy(Protocol):
    """Minimal protocol for models that provide grad_energy."""

    def grad_energy(self, x: jax.Array) -> jax.Array:
        """Gradient of energy w.r.t. x."""
        ...


def dsm_loss(
    grad_energy_fn: HasGradEnergy | callable,
    batch: jax.Array,
    noise: jax.Array,
    sigma: float,
) -> jax.Array:
    """Compute denoising score matching loss.

    Args:
        grad_energy_fn: Either an object with grad_energy method,
            or a callable (x -> grad_energy).
        batch: Data batch of shape (batch_size, dim).
        noise: Pre-sampled noise of shape (batch_size, dim).
        sigma: Noise standard deviation.

    Returns:
        Scalar DSM loss.

    Spec: REQ-TRAIN-002
    """
    sigma_sq = sigma * sigma
    noisy_batch = batch + noise
    target = -noise / sigma_sq

    # Support both protocol objects and raw callables
    if callable(grad_energy_fn) and not isinstance(grad_energy_fn, HasGradEnergy):
        grad_fn = grad_energy_fn
    else:
        grad_fn = grad_energy_fn.grad_energy

    # Vectorize over batch dimension
    model_scores = jax.vmap(grad_fn)(noisy_batch)
    diff = model_scores - target
    per_sample_loss = jnp.sum(diff**2, axis=-1)
    return jnp.mean(per_sample_loss)


def dsm_loss_stochastic(
    grad_energy_fn: HasGradEnergy | callable,
    batch: jax.Array,
    sigma: float,
    key: jax.Array,
) -> jax.Array:
    """Compute DSM loss with freshly sampled noise.

    Args:
        grad_energy_fn: Model or callable providing grad_energy.
        batch: Data batch of shape (batch_size, dim).
        sigma: Noise standard deviation.
        key: JAX PRNG key for noise sampling.

    Returns:
        Scalar DSM loss.

    Spec: REQ-TRAIN-002
    """
    noise = jax.random.normal(key, shape=batch.shape) * sigma
    return dsm_loss(grad_energy_fn, batch, noise, sigma)
