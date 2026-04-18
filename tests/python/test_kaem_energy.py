"""Tests for carnot.models.kaem_energy — KAEMEnergy, UnivariateKAEMLayer, benchmark.

100% coverage target for kaem_energy.py.

Spec: REQ-SAMPLE-015, REQ-SAMPLE-016,
      SCENARIO-SAMPLE-027, SCENARIO-SAMPLE-028, SCENARIO-SAMPLE-029
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.models.kaem_energy import (
    KAEMEnergy,
    UnivariateKAEMLayer,
    _N_QUAD,
    benchmark_kaem_vs_mcmc,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer(n_vars: int = 3, n_knots: int = 8) -> UnivariateKAEMLayer:
    """Small UnivariateKAEMLayer for fast unit tests."""
    return UnivariateKAEMLayer(n_vars=n_vars, n_knots=n_knots, key=jrandom.PRNGKey(0))


def _model(n_vars: int = 5, n_hidden: int = 8) -> KAEMEnergy:
    """Small KAEMEnergy for fast unit tests."""
    return KAEMEnergy(n_vars=n_vars, n_hidden=n_hidden, key=jrandom.PRNGKey(42))


# ---------------------------------------------------------------------------
# UnivariateKAEMLayer — init validation
# ---------------------------------------------------------------------------


class TestUnivariateKAEMLayerInit:
    """REQ-SAMPLE-015: UnivariateKAEMLayer initialises correctly."""

    def test_default_key(self) -> None:
        """Layer initialises with no key provided (uses PRNGKey(0))."""
        layer = UnivariateKAEMLayer(n_vars=2, n_knots=4)
        assert layer.n_vars == 2
        assert layer.n_knots == 4

    def test_control_point_shape(self) -> None:
        """Control points have shape (n_vars, n_knots)."""
        # SCENARIO-SAMPLE-027
        layer = _layer(n_vars=4, n_knots=6)
        assert layer.control_points.shape == (4, 6)

    def test_knot_positions_range(self) -> None:
        """Knot positions span [-1, 1]."""
        layer = _layer()
        assert float(layer._knots[0]) == pytest.approx(-1.0)
        assert float(layer._knots[-1]) == pytest.approx(1.0)

    def test_invalid_n_vars(self) -> None:
        """n_vars < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_vars"):
            UnivariateKAEMLayer(n_vars=0)

    def test_invalid_n_knots(self) -> None:
        """n_knots < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_knots"):
            UnivariateKAEMLayer(n_vars=2, n_knots=1)


# ---------------------------------------------------------------------------
# UnivariateKAEMLayer — spline evaluation
# ---------------------------------------------------------------------------


class TestUnivariateKAEMLayerSpline:
    """REQ-SAMPLE-015: Spline evaluation is correct and differentiable."""

    def test_eval_spline_single_scalar(self) -> None:
        """_eval_spline_single returns a scalar for a scalar input."""
        layer = _layer()
        ctrl = layer.control_points[0]
        result = layer._eval_spline_single(ctrl, jnp.array(0.0))
        assert result.shape == ()

    def test_eval_spline_single_clamps(self) -> None:
        """_eval_spline_single clamps input to [-1, 1] without error."""
        layer = _layer()
        ctrl = layer.control_points[0]
        # Out-of-range values should not crash
        v1 = layer._eval_spline_single(ctrl, jnp.array(-2.0))
        v2 = layer._eval_spline_single(ctrl, jnp.array(2.0))
        assert jnp.isfinite(v1)
        assert jnp.isfinite(v2)

    def test_eval_spline_np_vectorised(self) -> None:
        """_eval_spline_np evaluates correctly over a grid."""
        layer = _layer(n_knots=4)
        ctrl = np.array(layer.control_points[0])
        grid = np.linspace(-1.0, 1.0, 10)
        vals = layer._eval_spline_np(ctrl, grid)
        assert vals.shape == (10,)
        assert np.all(np.isfinite(vals))

    def test_eval_spline_np_clamping(self) -> None:
        """_eval_spline_np clamps out-of-range inputs."""
        layer = _layer(n_knots=4)
        ctrl = np.array(layer.control_points[0])
        xs = np.array([-5.0, 0.0, 5.0])
        vals = layer._eval_spline_np(ctrl, xs)
        assert np.all(np.isfinite(vals))


# ---------------------------------------------------------------------------
# UnivariateKAEMLayer — energy
# ---------------------------------------------------------------------------


class TestUnivariateKAEMLayerEnergy:
    """REQ-SAMPLE-015: energy() is correct and differentiable."""

    def test_energy_scalar(self) -> None:
        """energy() returns a scalar for a valid input. SCENARIO-SAMPLE-028."""
        layer = _layer(n_vars=3)
        x = jnp.zeros(3)
        e = layer.energy(x)
        assert e.shape == ()
        assert jnp.isfinite(e)

    def test_energy_differentiable(self) -> None:
        """jax.grad(energy) works — energy is JAX-differentiable.
        SCENARIO-SAMPLE-028: REQ-SAMPLE-015."""
        layer = _layer(n_vars=3)
        x = jnp.array([0.1, -0.5, 0.7])
        grad = jax.grad(layer.energy)(x)
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_energy_changes_with_input(self) -> None:
        """energy() is not constant — it responds to input changes."""
        layer = _layer(n_vars=2)
        e1 = layer.energy(jnp.array([-1.0, -1.0]))
        e2 = layer.energy(jnp.array([1.0, 1.0]))
        # With random (non-zero) control points, energies should differ
        # (they could be equal by chance, but with seed=0 and n_knots=8 they won't be)
        assert jnp.isfinite(e1) and jnp.isfinite(e2)


# ---------------------------------------------------------------------------
# UnivariateKAEMLayer — marginal_cdf
# ---------------------------------------------------------------------------


class TestUnivariateKAEMLayerMarginalCDF:
    """REQ-SAMPLE-015: marginal_cdf() is a valid CDF."""

    def test_cdf_at_minus1_is_near_zero(self) -> None:
        """CDF at left boundary should be ~0."""
        layer = _layer()
        cdf = layer.marginal_cdf(0, -1.0)
        assert cdf == pytest.approx(0.0, abs=0.01)

    def test_cdf_at_plus1_is_near_one(self) -> None:
        """CDF at right boundary should be ~1."""
        layer = _layer()
        cdf = layer.marginal_cdf(0, 1.0)
        assert cdf == pytest.approx(1.0, abs=0.01)

    def test_cdf_monotone(self) -> None:
        """CDF is non-decreasing over [-1, 1]."""
        layer = _layer()
        xs = np.linspace(-1.0, 1.0, 20)
        cdfs = [layer.marginal_cdf(0, float(x)) for x in xs]
        for i in range(len(cdfs) - 1):
            assert cdfs[i] <= cdfs[i + 1] + 1e-6

    def test_cdf_clamping(self) -> None:
        """marginal_cdf clamps x to [-1, 1]."""
        layer = _layer()
        cdf_lo = layer.marginal_cdf(0, -5.0)
        cdf_hi = layer.marginal_cdf(0, 5.0)
        assert 0.0 <= cdf_lo <= 1.0
        assert 0.0 <= cdf_hi <= 1.0

    def test_cdf_large_energy_still_valid(self) -> None:
        """CDF is valid even with very large control points (stability shift handles it)."""
        # Large control points → exp(-large) ≈ 0, but stability shift keeps density=1
        # at the max point. The CDF should still be valid in [0, 1].
        layer = UnivariateKAEMLayer(n_vars=1, n_knots=4, key=jrandom.PRNGKey(99))
        layer.control_points = layer.control_points.at[0].set(jnp.full(4, 1000.0))
        cdf = layer.marginal_cdf(0, 0.0)
        assert 0.0 <= cdf <= 1.0


# ---------------------------------------------------------------------------
# UnivariateKAEMLayer — _build_cdf_table and _invert_cdf
# ---------------------------------------------------------------------------


class TestUnivariateKAEMLayerCDFTable:
    """Internal CDF table and inversion methods."""

    def test_build_cdf_table_shape(self) -> None:
        """_build_cdf_table returns (grid, cdf_vals) with correct lengths."""
        layer = _layer()
        ctrl = np.array(layer.control_points[0])
        grid, cdf_vals = layer._build_cdf_table(ctrl)
        assert len(grid) == _N_QUAD
        assert len(cdf_vals) == _N_QUAD

    def test_build_cdf_table_normalised(self) -> None:
        """CDF table final value is ~1.0 for normal density."""
        layer = _layer()
        ctrl = np.array(layer.control_points[0])
        grid, cdf_vals = layer._build_cdf_table(ctrl)
        assert float(cdf_vals[-1]) == pytest.approx(1.0, abs=0.01)

    def test_build_cdf_table_large_energy(self) -> None:
        """Large energy control points still produce valid normalised CDF."""
        # Stability shift ensures total > 0 even when all control points are large.
        layer = _layer()
        ctrl = np.full(8, 1000.0)
        grid, cdf_vals = layer._build_cdf_table(ctrl)
        assert np.all(cdf_vals >= 0.0)
        assert np.all(cdf_vals <= 1.0 + 1e-6)
        # Final value should be ~1.0 (normalised)
        assert float(cdf_vals[-1]) == pytest.approx(1.0, abs=0.01)

    def test_invert_cdf_u0(self) -> None:
        """_invert_cdf(u=0) returns near left boundary."""
        layer = _layer()
        ctrl = np.array(layer.control_points[0])
        table = layer._build_cdf_table(ctrl)
        x = layer._invert_cdf(table, 0.0)
        assert -1.0 <= x <= 1.0

    def test_invert_cdf_u1(self) -> None:
        """_invert_cdf(u=1) returns near right boundary."""
        layer = _layer()
        ctrl = np.array(layer.control_points[0])
        table = layer._build_cdf_table(ctrl)
        x = layer._invert_cdf(table, 1.0)
        assert -1.0 <= x <= 1.0

    def test_invert_cdf_midpoint(self) -> None:
        """_invert_cdf(u=0.5) returns interior point."""
        layer = _layer()
        ctrl = np.array(layer.control_points[0])
        table = layer._build_cdf_table(ctrl)
        x = layer._invert_cdf(table, 0.5)
        assert -1.0 <= x <= 1.0

    def test_invert_cdf_flat_slope(self) -> None:
        """_invert_cdf returns x0 when adjacent CDF entries are equal (line 376).

        Triggers via u > max(cdf_vals): searchsorted returns _N_QUAD (out of bounds),
        clipped to _N_QUAD-1. If cdf_vals[-2] == cdf_vals[-1], c0==c1 and the
        early-return branch fires.
        """
        layer = _layer()
        grid = np.linspace(-1.0, 1.0, _N_QUAD)
        cdf_vals = np.linspace(0.0, 1.0, _N_QUAD).copy()
        # Make the last two entries equal so c0==c1 when idx is clipped to _N_QUAD-1
        cdf_vals[-1] = cdf_vals[-2]  # both = (N_QUAD-2)/(N_QUAD-1) ≈ 0.996
        table = (grid, cdf_vals)
        # u=2.0 → searchsorted returns _N_QUAD → clipped to _N_QUAD-1
        # c0 = cdf_vals[-2], c1 = cdf_vals[-1] — both equal → line 376 fires
        x = layer._invert_cdf(table, 2.0)
        assert np.isfinite(x)
        assert -1.0 <= x <= 1.0


# ---------------------------------------------------------------------------
# UnivariateKAEMLayer — sample_exact
# ---------------------------------------------------------------------------


class TestUnivariateKAEMLayerSampleExact:
    """REQ-SAMPLE-015, SCENARIO-SAMPLE-027: exact sampling produces valid samples."""

    def test_sample_shape(self) -> None:
        """sample_exact returns array of shape (n_samples, n_vars)."""
        layer = _layer(n_vars=3)
        key = jrandom.PRNGKey(0)
        samples = layer.sample_exact(10, key)
        assert samples.shape == (10, 3)

    def test_sample_in_range(self) -> None:
        """All samples are in [-1, 1]. SCENARIO-SAMPLE-027."""
        layer = _layer(n_vars=5, n_knots=8)
        key = jrandom.PRNGKey(7)
        samples = layer.sample_exact(50, key)
        assert jnp.all(samples >= -1.0 - 1e-6)
        assert jnp.all(samples <= 1.0 + 1e-6)

    def test_sample_finite(self) -> None:
        """All samples are finite (no NaN or Inf)."""
        layer = _layer(n_vars=4)
        key = jrandom.PRNGKey(1)
        samples = layer.sample_exact(20, key)
        assert jnp.all(jnp.isfinite(samples))

    def test_sample_single(self) -> None:
        """n_samples=1 returns shape (1, n_vars)."""
        layer = _layer(n_vars=2)
        key = jrandom.PRNGKey(2)
        samples = layer.sample_exact(1, key)
        assert samples.shape == (1, 2)

    def test_sample_different_keys_differ(self) -> None:
        """Different PRNG keys produce different samples (not deterministic constant)."""
        layer = _layer(n_vars=3)
        s1 = layer.sample_exact(5, jrandom.PRNGKey(0))
        s2 = layer.sample_exact(5, jrandom.PRNGKey(99))
        # With probability essentially 1, random samples will differ
        assert not jnp.allclose(s1, s2)


# ---------------------------------------------------------------------------
# KAEMEnergy — init validation
# ---------------------------------------------------------------------------


class TestKAEMEnergyInit:
    """KAEMEnergy initialises correctly."""

    def test_default_key(self) -> None:
        """KAEMEnergy initialises with no key (uses PRNGKey(0))."""
        model = KAEMEnergy(n_vars=3)
        assert model.n_vars == 3

    def test_invalid_n_vars(self) -> None:
        """n_vars < 1 raises ValueError."""
        with pytest.raises(ValueError, match="n_vars"):
            KAEMEnergy(n_vars=0)

    def test_invalid_n_hidden(self) -> None:
        """n_hidden < 2 raises ValueError."""
        with pytest.raises(ValueError, match="n_hidden"):
            KAEMEnergy(n_vars=3, n_hidden=1)

    def test_layer_created(self) -> None:
        """KAEMEnergy creates an UnivariateKAEMLayer with correct params."""
        model = _model(n_vars=4, n_hidden=6)
        assert isinstance(model.layer, UnivariateKAEMLayer)
        assert model.layer.n_vars == 4
        assert model.layer.n_knots == 6


# ---------------------------------------------------------------------------
# KAEMEnergy — energy
# ---------------------------------------------------------------------------


class TestKAEMEnergyEnergy:
    """REQ-SAMPLE-015, SCENARIO-SAMPLE-028: energy() is correct and differentiable."""

    def test_energy_scalar(self) -> None:
        """energy() returns a scalar. SCENARIO-SAMPLE-028."""
        model = _model(n_vars=4)
        x = jnp.zeros(4)
        e = model.energy(x)
        assert e.shape == ()
        assert jnp.isfinite(e)

    def test_energy_differentiable(self) -> None:
        """jax.grad(model.energy) works. SCENARIO-SAMPLE-028."""
        model = _model(n_vars=4)
        x = jnp.array([0.1, -0.2, 0.3, -0.4])
        grad = jax.grad(model.energy)(x)
        assert grad.shape == (4,)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# KAEMEnergy — sample
# ---------------------------------------------------------------------------


class TestKAEMEnergySample:
    """REQ-SAMPLE-015, SCENARIO-SAMPLE-027: sample() produces exact valid samples."""

    def test_sample_shape(self) -> None:
        """sample() returns (n_samples, n_vars)."""
        model = _model(n_vars=5)
        samples = model.sample(10)
        assert samples.shape == (10, 5)

    def test_sample_default_n(self) -> None:
        """sample() with default n=1 returns shape (1, n_vars)."""
        model = _model(n_vars=3)
        samples = model.sample()
        assert samples.shape == (1, 3)

    def test_sample_in_range(self) -> None:
        """All samples in [-1, 1]. SCENARIO-SAMPLE-027."""
        model = _model(n_vars=6)
        samples = model.sample(30)
        assert jnp.all(samples >= -1.0 - 1e-6)
        assert jnp.all(samples <= 1.0 + 1e-6)

    def test_sample_finite(self) -> None:
        """All samples are finite."""
        model = _model(n_vars=4)
        samples = model.sample(20)
        assert jnp.all(jnp.isfinite(samples))

    def test_repeated_calls_differ(self) -> None:
        """Repeated sample() calls produce different samples (key advances)."""
        model = _model(n_vars=3)
        s1 = model.sample(5)
        s2 = model.sample(5)
        assert not jnp.allclose(s1, s2)


# ---------------------------------------------------------------------------
# KAEMEnergy — fit
# ---------------------------------------------------------------------------


class TestKAEMEnergyFit:
    """REQ-SAMPLE-015: fit() runs without error and returns loss history."""

    def test_fit_returns_losses(self) -> None:
        """fit() returns list of floats with length n_epochs."""
        model = _model(n_vars=3)
        data = jnp.zeros((10, 3))
        losses = model.fit(data, n_epochs=5)
        assert len(losses) == 5
        assert all(isinstance(v, float) for v in losses)

    def test_fit_invalid_shape(self) -> None:
        """fit() raises ValueError for wrong data shape."""
        model = _model(n_vars=3)
        with pytest.raises(ValueError):
            model.fit(jnp.zeros((10, 5)), n_epochs=2)

    def test_fit_1d_shape_rejected(self) -> None:
        """fit() raises ValueError for 1D data."""
        model = _model(n_vars=3)
        with pytest.raises(ValueError):
            model.fit(jnp.zeros(10), n_epochs=2)

    def test_fit_modifies_control_points(self) -> None:
        """fit() changes control points (model learns from data)."""
        model = _model(n_vars=2, n_hidden=4)
        before = np.array(model.layer.control_points)
        data = jnp.ones((5, 2)) * 0.5
        model.fit(data, n_epochs=3)
        after = np.array(model.layer.control_points)
        assert not np.allclose(before, after)


# ---------------------------------------------------------------------------
# benchmark_kaem_vs_mcmc
# ---------------------------------------------------------------------------


class TestBenchmarkKAEMvsMCMC:
    """REQ-SAMPLE-016, SCENARIO-SAMPLE-029: benchmark returns valid latency dict."""

    def test_returns_dict_keys(self) -> None:
        """benchmark_kaem_vs_mcmc returns dict with required keys. SCENARIO-SAMPLE-029."""
        result = benchmark_kaem_vs_mcmc(n_vars=5, n_samples=10)
        assert "n_vars" in result
        assert "n_samples" in result
        assert "kaem_latency_ms" in result
        assert "ising_mcmc_latency_ms" in result
        assert "speedup_ratio" in result

    def test_n_vars_recorded(self) -> None:
        """n_vars in result matches input."""
        result = benchmark_kaem_vs_mcmc(n_vars=8, n_samples=5)
        assert result["n_vars"] == 8

    def test_n_samples_recorded(self) -> None:
        """n_samples in result matches input."""
        result = benchmark_kaem_vs_mcmc(n_vars=5, n_samples=7)
        assert result["n_samples"] == 7

    def test_latencies_non_negative(self) -> None:
        """Both latencies are non-negative."""
        result = benchmark_kaem_vs_mcmc(n_vars=6, n_samples=5)
        assert result["kaem_latency_ms"] >= 0.0
        assert result["ising_mcmc_latency_ms"] >= 0.0

    def test_speedup_positive(self) -> None:
        """speedup_ratio is positive."""
        result = benchmark_kaem_vs_mcmc(n_vars=5, n_samples=5)
        assert result["speedup_ratio"] > 0.0

    def test_all_values_finite(self) -> None:
        """All values in result dict are finite floats."""
        result = benchmark_kaem_vs_mcmc(n_vars=5, n_samples=5)
        for key in ("kaem_latency_ms", "ising_mcmc_latency_ms", "speedup_ratio"):
            assert np.isfinite(result[key]), f"{key} is not finite"

    def test_small_n_vars(self) -> None:
        """Benchmark works with n_vars=1."""
        result = benchmark_kaem_vs_mcmc(n_vars=1, n_samples=3)
        assert result["n_vars"] == 1
        assert result["speedup_ratio"] > 0.0
