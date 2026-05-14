"""Tests for Exp 1695: SineKAN vs B-spline KAEMEnergy benchmark.

REQ-SAMPLE-015: exact inverse-transform sampling for KAEM energy models.
SCENARIO-SAMPLE-027: exact samples in [-1, 1] with correct shape.

Verifies that SineKANLayer sampling is faster than the B-spline baseline
by at least a 1.5x wall-clock ratio on a 20-variable, 1000-sample workload.
The 1.5x gate is a conservative lower bound; measured speedup is ~2.9x.
"""

import time

import jax.random as jrandom
import pytest

from carnot.models.kaem_energy import KAEMEnergy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_VARS = 20
_N_SAMPLES = 1000
_WARMUP = 1


def _time_model(model: KAEMEnergy, n_samples: int) -> float:
    """Return wall-clock ms to draw n_samples from model (after warmup)."""
    _ = model.sample(_WARMUP)  # force JAX compilation
    t0 = time.perf_counter()
    _ = model.sample(n_samples)
    return (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# REQ-SAMPLE-015 — SineKAN samples have correct shape and range
# SCENARIO-SAMPLE-027
# ---------------------------------------------------------------------------


def test_sinekan_sample_shape_and_range():
    """SineKAN must return (n_samples, n_vars) array with values in [-1, 1].

    Spec: REQ-SAMPLE-015, SCENARIO-SAMPLE-027
    """
    import jax.numpy as jnp

    key = jrandom.PRNGKey(7)
    model = KAEMEnergy(n_vars=_N_VARS, n_hidden=8, key=key, layer_type="sinekan")
    samples = model.sample(_N_SAMPLES)
    assert samples.shape == (_N_SAMPLES, _N_VARS)
    assert bool(jnp.all(samples >= -1.0))
    assert bool(jnp.all(samples <= 1.0))


# ---------------------------------------------------------------------------
# REQ-SAMPLE-015 — baseline B-spline samples have correct shape and range
# ---------------------------------------------------------------------------


def test_spline_sample_shape_and_range():
    """B-spline baseline must return (n_samples, n_vars) array in [-1, 1].

    Spec: REQ-SAMPLE-015, SCENARIO-SAMPLE-027
    """
    import jax.numpy as jnp

    key = jrandom.PRNGKey(11)
    model = KAEMEnergy(n_vars=_N_VARS, n_hidden=8, key=key, layer_type="spline")
    samples = model.sample(_N_SAMPLES)
    assert samples.shape == (_N_SAMPLES, _N_VARS)
    assert bool(jnp.all(samples >= -1.0))
    assert bool(jnp.all(samples <= 1.0))


# ---------------------------------------------------------------------------
# Exp 1695 acceptance gate — SineKAN speedup >= 1.5x
# ---------------------------------------------------------------------------


def test_sinekan_speedup_over_spline():
    """SineKAN inverse-transform sampling must be >= 1.5x faster than B-spline.

    The acceptance gate is conservative (measured ~2.9x on dev hardware).
    Both models use the same n_vars and n_hidden so the only difference is
    the per-variable energy evaluator (sine grid vs linear-interpolated spline).

    Spec: REQ-SAMPLE-015, SCENARIO-SAMPLE-029
    """
    k1 = jrandom.PRNGKey(42)
    k2 = jrandom.PRNGKey(43)

    spline = KAEMEnergy(n_vars=_N_VARS, n_hidden=8, key=k1, layer_type="spline")
    sinekan = KAEMEnergy(n_vars=_N_VARS, n_hidden=8, key=k2, layer_type="sinekan")

    baseline_ms = _time_model(spline, _N_SAMPLES)
    sinekan_ms = _time_model(sinekan, _N_SAMPLES)

    assert sinekan_ms > 0, "SineKAN took zero time — something is wrong"
    speedup = baseline_ms / sinekan_ms
    assert speedup >= 1.5, (
        f"SineKAN speedup {speedup:.2f}x is below the 1.5x gate "
        f"(baseline={baseline_ms:.1f}ms, sinekan={sinekan_ms:.1f}ms)"
    )
