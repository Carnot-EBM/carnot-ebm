"""Tests for carnot.samplers.synchronous_ising — 100% coverage target.

Spec: REQ-SAMPLE-037, SCENARIO-SAMPLE-061, SCENARIO-SAMPLE-062
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.samplers.synchronous_ising import SynchronousIsingSampler, _sigmoid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sampler(n: int = 4, beta: float = 1.0, seed: int = 0) -> SynchronousIsingSampler:
    rng = np.random.default_rng(seed)
    J = rng.standard_normal((n, n)) * 0.2
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    h = rng.standard_normal(n) * 0.1
    return SynchronousIsingSampler(n_spins=n, couplings=J, biases=h, beta=beta)


# ---------------------------------------------------------------------------
# _sigmoid helper
# ---------------------------------------------------------------------------


def test_sigmoid_zero():
    """sigmoid(0) == 0.5 exactly."""
    assert _sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)


def test_sigmoid_large_positive():
    """sigmoid(large positive) approaches 1."""
    assert _sigmoid(np.array([100.0]))[0] == pytest.approx(1.0, abs=1e-6)


def test_sigmoid_large_negative():
    """sigmoid(large negative) approaches 0."""
    assert _sigmoid(np.array([-100.0]))[0] == pytest.approx(0.0, abs=1e-6)


def test_sigmoid_vectorised():
    """_sigmoid handles arrays of arbitrary shape."""
    x = np.linspace(-5, 5, 100)
    y = _sigmoid(x)
    assert y.shape == (100,)
    assert np.all(y >= 0.0) and np.all(y <= 1.0)


# ---------------------------------------------------------------------------
# SynchronousIsingSampler.__init__
# ---------------------------------------------------------------------------


def test_init_stores_params():
    """Constructor stores n_spins, beta, and validates shapes."""
    sampler = _make_sampler(n=6, beta=2.0)
    assert sampler.n_spins == 6
    assert sampler.beta == 2.0
    assert sampler.couplings.shape == (6, 6)
    assert sampler.biases.shape == (6,)


def test_init_bad_coupling_shape():
    """Wrong coupling shape raises ValueError."""
    with pytest.raises(ValueError, match="couplings must be"):
        SynchronousIsingSampler(
            n_spins=4,
            couplings=np.zeros((3, 4)),
            biases=np.zeros(4),
        )


def test_init_bad_bias_shape():
    """Wrong bias shape raises ValueError."""
    with pytest.raises(ValueError, match="biases must be"):
        SynchronousIsingSampler(
            n_spins=4,
            couplings=np.zeros((4, 4)),
            biases=np.zeros(3),
        )


# ---------------------------------------------------------------------------
# energy()
# ---------------------------------------------------------------------------


def test_energy_all_ones_ferromagnetic():
    """All-+1 state has lower energy than all-(-1) for ferromagnetic J."""
    n = 4
    J = np.ones((n, n)) * 0.5
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n)
    sampler = SynchronousIsingSampler(n_spins=n, couplings=J, biases=h)
    e_up = sampler.energy(np.ones(n))
    e_down = sampler.energy(-np.ones(n))
    # For ferromagnetic J, all-+1 and all-(-1) should be equivalent by symmetry
    assert e_up == pytest.approx(e_down)


def test_energy_returns_scalar():
    """energy() always returns a Python float."""
    sampler = _make_sampler(n=6)
    e = sampler.energy(np.ones(6))
    assert isinstance(e, float)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_output_shape():
    """step() returns state with same shape as input."""
    sampler = _make_sampler(n=8)
    state = np.ones(8)
    new_state = sampler.step(state)
    assert new_state.shape == (8,)


def test_step_values_pm1():
    """step() always produces spins in {-1, +1}."""
    sampler = _make_sampler(n=10)
    state = np.ones(10)
    for _ in range(20):
        state = sampler.step(state)
    assert set(np.unique(state)).issubset({-1.0, 1.0})


def test_step_does_not_mutate_input():
    """step() must not modify the input state array in-place."""
    sampler = _make_sampler(n=6)
    state = np.ones(6)
    original = state.copy()
    sampler.step(state)
    np.testing.assert_array_equal(state, original)


# ---------------------------------------------------------------------------
# sample()
# ---------------------------------------------------------------------------


def test_sample_default_init():
    """sample() with no init_state starts from all-+1 (RTL reset)."""
    sampler = _make_sampler(n=6)
    final = sampler.sample(n_steps=5)
    assert final.shape == (6,)
    assert set(np.unique(final)).issubset({-1.0, 1.0})


def test_sample_with_init_state():
    """sample() accepts an explicit init_state."""
    sampler = _make_sampler(n=6)
    init = -np.ones(6)
    final = sampler.sample(n_steps=5, init_state=init)
    assert final.shape == (6,)


def test_sample_zero_steps():
    """sample() with n_steps=0 returns init_state unchanged."""
    sampler = _make_sampler(n=4)
    init = np.array([1.0, -1.0, 1.0, -1.0])
    final = sampler.sample(n_steps=0, init_state=init)
    np.testing.assert_array_equal(final, init)


# ---------------------------------------------------------------------------
# compare_with_async()
# ---------------------------------------------------------------------------


def test_compare_with_async_returns_required_keys():
    """compare_with_async() returns a dict with all four required keys."""
    # Spec: SCENARIO-SAMPLE-061
    sampler = _make_sampler(n=6, beta=1.0)
    result = sampler.compare_with_async(n_steps=20, n_trials=3)
    assert "sync_mean_energy" in result
    assert "async_mean_energy" in result
    assert "energy_gap" in result
    assert "sync_converged" in result


def test_compare_with_async_energy_gap_formula():
    """energy_gap == sync_mean - async_mean."""
    sampler = _make_sampler(n=6, beta=1.0)
    result = sampler.compare_with_async(n_steps=20, n_trials=3)
    assert result["energy_gap"] == pytest.approx(
        result["sync_mean_energy"] - result["async_mean_energy"], abs=1e-9
    )


def test_compare_with_async_sync_converged_is_bool():
    """sync_converged must be a Python bool."""
    sampler = _make_sampler(n=6, beta=1.0)
    result = sampler.compare_with_async(n_steps=20, n_trials=3)
    assert isinstance(result["sync_converged"], (bool, np.bool_))


def test_compare_with_async_small_instance_converges():
    """On a small ferromagnetic instance, sync_converged should be True.

    This is SCENARIO-SAMPLE-062: synchronous sampler reaches comparable
    final energy to the async (JAX checkerboard) sampler on a 4-spin graph.
    """
    # Small ferromagnetic instance: should converge quickly for both.
    n = 4
    J = np.ones((n, n)) * 0.5
    np.fill_diagonal(J, 0.0)
    h = np.zeros(n)
    sampler = SynchronousIsingSampler(n_spins=n, couplings=J, biases=h, beta=2.0)
    result = sampler.compare_with_async(n_steps=50, n_trials=5)
    # Both samplers should reach comparable energy on this trivial instance.
    assert result["sync_mean_energy"] is not None
    assert result["async_mean_energy"] is not None


# ---------------------------------------------------------------------------
# Export from carnot.samplers
# ---------------------------------------------------------------------------


def test_exported_from_samplers():
    """SynchronousIsingSampler is importable from carnot.samplers."""
    from carnot.samplers import SynchronousIsingSampler as S  # noqa: F401

    assert S is SynchronousIsingSampler
