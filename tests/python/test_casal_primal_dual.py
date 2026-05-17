"""Tests for Exp 2245 CASAL primal-dual equality sampler.

Spec coverage: REQ-SAMPLE-2245, REQ-SAMPLE-2245-1, REQ-SAMPLE-2245-2,
REQ-SAMPLE-2245-3, REQ-SAMPLE-2245-4, SCENARIO-SAMPLE-2245.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from carnot.samplers.casal import CASALSampler


def test_casal_sampler_enforces_hard_equality_constraint() -> None:
    """SCENARIO-SAMPLE-2245: Mean violation stays below the hard-constraint gate."""

    def energy_fn(x: jax.Array) -> jax.Array:
        target = jnp.array([2.0, -1.0], dtype=x.dtype)
        return 0.5 * jnp.sum((x - target) ** 2)

    def equality_residual(x: jax.Array) -> jax.Array:
        return x[0] + x[1] - 1.0

    sampler = CASALSampler(
        constraints=[equality_residual],
        step_size=0.02,
        dual_step_size=0.8,
        n_steps=80,
        seed=2245,
    )
    sample = sampler.sample(jnp.array([4.0, -3.0]), energy_fn)

    final_violation = float(jnp.abs(equality_residual(sample)))
    assert jnp.all(jnp.isfinite(sample))
    assert final_violation < 1e-4
    assert sampler.last_violation_mean is not None
    assert sampler.last_violation_mean < 1e-4
    assert sampler.dual_update_converged()
