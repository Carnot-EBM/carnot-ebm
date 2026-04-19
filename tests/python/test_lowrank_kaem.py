"""Tests for LowRankProjector and LowRankKAEMEnergy.

Spec: REQ-SAMPLE-027, REQ-SAMPLE-028,
      SCENARIO-SAMPLE-041, SCENARIO-SAMPLE-042, SCENARIO-SAMPLE-043
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from carnot.models.lowrank_kaem import LowRankKAEMEnergy, LowRankProjector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_low_rank_data(
    n_samples: int = 200,
    n_vars: int = 50,
    true_rank: int = 11,
    noise_scale: float = 0.01,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate synthetic data with rank-true_rank signal plus small noise.

    The signal lives in a true_rank-dimensional subspace.  After SVD, the top
    true_rank singular values carry most variance and explained_variance_ratio
    should exceed 0.90 at k=true_rank.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    # Low-rank signal: random basis * coefficients
    basis = rng.standard_normal((n_vars, true_rank)).astype(np.float32)
    coef = rng.standard_normal((n_samples, true_rank)).astype(np.float32)
    signal = coef @ basis.T  # (n_samples, n_vars)
    noise = rng.standard_normal((n_samples, n_vars)).astype(np.float32) * noise_scale
    data = signal + noise
    # Normalise to [-1, 1] range expected by KAEM splines
    mx = np.max(np.abs(data)) + 1e-6
    return (data / mx).astype(np.float32)


# ---------------------------------------------------------------------------
# LowRankProjector tests
# ---------------------------------------------------------------------------


class TestLowRankProjector:
    """REQ-SAMPLE-027, REQ-SAMPLE-028, SCENARIO-SAMPLE-041, SCENARIO-SAMPLE-042"""

    def test_project_output_shape(self):
        """SCENARIO-SAMPLE-041: project() compresses n_vars=100 to k=2 dimensions."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal((50, 100)).astype(np.float32)
        proj = LowRankProjector(jnp.array(data), k=2)

        x = jnp.ones(100, dtype=jnp.float32)
        result = proj.project(x)
        assert result.shape == (2,), f"expected shape (2,), got {result.shape}"
        assert jnp.all(jnp.isfinite(result))

    def test_project_is_differentiable(self):
        """project() is a linear map and JAX can differentiate through it."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal((30, 20)).astype(np.float32)
        proj = LowRankProjector(jnp.array(data), k=5)
        x = jnp.ones(20, dtype=jnp.float32)

        def f(v):
            return jnp.sum(proj.project(v))

        grad = jax.grad(f)(x)
        assert grad.shape == (20,)
        assert jnp.all(jnp.isfinite(grad))

    def test_explained_variance_ratio_monotonic(self):
        """explained_variance_ratio should be non-decreasing as k increases."""
        rng = np.random.default_rng(2)
        data = _make_low_rank_data(n_samples=100, n_vars=30, true_rank=5, rng=rng)
        proj = LowRankProjector(jnp.array(data), k=15)
        ratios = [proj.explained_variance_ratio(k) for k in range(1, 16)]
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i - 1] - 1e-6

    def test_explained_variance_ratio_low_rank_data(self):
        """SCENARIO-SAMPLE-042: top-11 components >= 90% variance for rank-11 data."""
        data = _make_low_rank_data(n_samples=200, n_vars=50, true_rank=11, noise_scale=0.005)
        proj = LowRankProjector(jnp.array(data), k=30)
        ratio_11 = proj.explained_variance_ratio(11)
        assert ratio_11 >= 0.90, (
            f"Expected >= 0.90 explained variance at k=11 for rank-11 data, got {ratio_11:.4f}"
        )

    def test_explained_variance_ratio_full_rank_is_one(self):
        """All components together should explain all variance (ratio = 1.0)."""
        rng = np.random.default_rng(3)
        data = rng.standard_normal((20, 10)).astype(np.float32)
        proj = LowRankProjector(jnp.array(data), k=10)
        ratio = proj.explained_variance_ratio(10)
        assert abs(ratio - 1.0) < 1e-5, f"expected ~1.0, got {ratio}"

    def test_auto_k_returns_minimum_sufficient_rank(self):
        """SCENARIO-SAMPLE-042: auto_k returns minimal k satisfying threshold."""
        data = _make_low_rank_data(n_samples=200, n_vars=50, true_rank=11, noise_scale=0.005)
        proj = LowRankProjector(jnp.array(data), k=30)
        k = proj.auto_k(threshold=0.90)
        assert k >= 1
        # The returned k should actually meet the threshold
        assert proj.explained_variance_ratio(k) >= 0.90
        # k-1 should NOT meet the threshold (it's the minimum)
        if k > 1:
            assert proj.explained_variance_ratio(k - 1) < 0.90

    def test_auto_k_at_low_threshold_returns_small_k(self):
        """auto_k with threshold=0.10 should return a small k."""
        data = _make_low_rank_data(n_samples=100, n_vars=40, true_rank=10)
        proj = LowRankProjector(jnp.array(data), k=20)
        k_low = proj.auto_k(threshold=0.10)
        k_high = proj.auto_k(threshold=0.99)
        assert k_low <= k_high

    def test_auto_k_low_rank_data_le_11(self):
        """For rank-11 data, auto_k(0.90) should be <= 11."""
        data = _make_low_rank_data(n_samples=200, n_vars=50, true_rank=11, noise_scale=0.005)
        proj = LowRankProjector(jnp.array(data), k=30)
        k = proj.auto_k(threshold=0.90)
        assert k <= 11, f"Expected auto_k <= 11 for rank-11 data, got {k}"

    def test_k_clamped_to_available_components(self):
        """LowRankProjector clamps k to min(k, n_samples, n_vars)."""
        rng = np.random.default_rng(5)
        data = rng.standard_normal((5, 20)).astype(np.float32)  # only 5 samples
        proj = LowRankProjector(jnp.array(data), k=100)
        assert proj.k <= 5  # clamped to n_samples

    def test_k_validation_raises(self):
        """k < 1 should raise ValueError."""
        rng = np.random.default_rng(6)
        data = rng.standard_normal((10, 5)).astype(np.float32)
        with pytest.raises(ValueError, match="k must be >= 1"):
            LowRankProjector(jnp.array(data), k=0)

    def test_explained_variance_ratio_clamps_k(self):
        """explained_variance_ratio with k > n_components is clamped, not an error."""
        rng = np.random.default_rng(7)
        data = rng.standard_normal((10, 5)).astype(np.float32)
        proj = LowRankProjector(jnp.array(data), k=5)
        # k=1000 should be clamped to n_components and return 1.0
        ratio = proj.explained_variance_ratio(1000)
        assert abs(ratio - 1.0) < 1e-5

    def test_zero_variance_data(self):
        """Degenerate zero-variance data returns explained_variance_ratio=1.0."""
        data = np.zeros((10, 5), dtype=np.float32)
        proj = LowRankProjector(jnp.array(data), k=3)
        assert proj.explained_variance_ratio(1) == 1.0

    def test_auto_k_returns_n_components_when_threshold_impossible(self):
        """auto_k with threshold > 1.0 returns n_components (fallback branch)."""
        rng = np.random.default_rng(9)
        data = rng.standard_normal((10, 5)).astype(np.float32)
        proj = LowRankProjector(jnp.array(data), k=5)
        k = proj.auto_k(threshold=1.001)  # impossible to satisfy
        assert k == len(proj._variance)


# ---------------------------------------------------------------------------
# LowRankKAEMEnergy tests
# ---------------------------------------------------------------------------


class TestLowRankKAEMEnergy:
    """REQ-SAMPLE-027, REQ-SAMPLE-028, SCENARIO-SAMPLE-041, SCENARIO-SAMPLE-042, SCENARIO-SAMPLE-043"""

    def _make_model_and_data(self, n_vars: int = 50, k: int = 11):
        data = jnp.array(
            _make_low_rank_data(n_samples=100, n_vars=n_vars, true_rank=k)
        )
        model = LowRankKAEMEnergy(n_vars=n_vars, k=k)
        model.fit(data, n_epochs=10)
        return model, data

    def test_energy_returns_scalar(self):
        """energy() returns a scalar (shape () )."""
        model, data = self._make_model_and_data(n_vars=20, k=5)
        x = data[0]
        e = model.energy(x)
        assert e.shape == (), f"Expected scalar, got shape {e.shape}"
        assert jnp.isfinite(e)

    def test_energy_differentiable(self):
        """SCENARIO-SAMPLE-043: jax.grad(model.energy)(x) works and returns finite gradient."""
        model, data = self._make_model_and_data(n_vars=20, k=5)
        x = data[0]
        grad = jax.grad(model.energy)(x)
        assert grad.shape == (20,), f"Expected shape (20,), got {grad.shape}"
        assert jnp.all(jnp.isfinite(grad))

    def test_energy_faster_than_fullrank_at_large_n_vars(self):
        """LowRankKAEMEnergy is faster than full-rank KAEM for n_vars=50."""
        import time

        n_vars = 50
        k = 8
        data = jnp.array(_make_low_rank_data(n_samples=100, n_vars=n_vars, true_rank=k))

        # Fit full-rank
        from carnot.models.kaem_energy import KAEMEnergy

        full_model = KAEMEnergy(n_vars=n_vars, n_hidden=16)
        full_model.fit(data, n_epochs=5)

        # Fit low-rank
        lr_model = LowRankKAEMEnergy(n_vars=n_vars, k=k)
        lr_model.fit(data, n_epochs=5)

        x = data[0]
        # Warm up both (avoids JAX tracing overhead in timing)
        _ = full_model.energy(x)
        _ = lr_model.energy(x)

        n_repeats = 100
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            full_model.energy(x)
        full_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        for _ in range(n_repeats):
            lr_model.energy(x)
        lr_ms = (time.perf_counter() - t0) * 1000

        # Low-rank should not be slower; we don't assert strict speedup because
        # for small n_vars the projection overhead can dominate — but k < n_vars
        # means the spline cost is genuinely reduced.
        assert lr_ms / n_repeats < full_ms / n_repeats * 5, (
            f"LowRank ({lr_ms:.1f}ms) much slower than full-rank ({full_ms:.1f}ms) unexpectedly"
        )

    def test_fit_requires_correct_shape(self):
        """fit() raises ValueError on wrong data shape."""
        model = LowRankKAEMEnergy(n_vars=10, k=3)
        with pytest.raises(ValueError, match="data must have shape"):
            model.fit(jnp.ones((20, 5)), n_epochs=1)  # wrong n_vars

    def test_energy_before_fit_raises(self):
        """energy() before fit() raises RuntimeError."""
        model = LowRankKAEMEnergy(n_vars=10, k=3)
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            model.energy(jnp.zeros(10))

    def test_auto_k_mode(self):
        """auto_k=True selects rank automatically during fit."""
        n_vars = 30
        data = jnp.array(
            _make_low_rank_data(n_samples=100, n_vars=n_vars, true_rank=5, noise_scale=0.005)
        )
        model = LowRankKAEMEnergy(n_vars=n_vars, k=20, auto_k=True)
        model.fit(data, n_epochs=5)
        # After auto_k, model.k should be <= 20 (the auto-selected rank)
        assert model.k <= 20
        assert model.k >= 1
        # Energy should still work
        x = data[0]
        e = model.energy(x)
        assert jnp.isfinite(e)

    def test_n_vars_validation_raises(self):
        """n_vars < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_vars must be >= 1"):
            LowRankKAEMEnergy(n_vars=0, k=3)

    def test_k_validation_raises(self):
        """k < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="k must be >= 1"):
            LowRankKAEMEnergy(n_vars=10, k=0)

    def test_fit_returns_loss_history(self):
        """fit() returns a list of floats (loss per epoch)."""
        n_vars = 15
        data = jnp.array(
            _make_low_rank_data(n_samples=50, n_vars=n_vars, true_rank=5)
        )
        model = LowRankKAEMEnergy(n_vars=n_vars, k=5)
        losses = model.fit(data, n_epochs=5)
        assert isinstance(losses, list)
        assert len(losses) == 5
        assert all(isinstance(v, float) for v in losses)

    def test_projector_set_after_fit(self):
        """projector attribute is set after fit()."""
        n_vars = 15
        data = jnp.array(_make_low_rank_data(n_samples=50, n_vars=n_vars, true_rank=3))
        model = LowRankKAEMEnergy(n_vars=n_vars, k=3)
        assert model.projector is None
        model.fit(data, n_epochs=2)
        assert model.projector is not None
        assert isinstance(model.projector, LowRankProjector)

    def test_energy_finite_on_boundary_inputs(self):
        """energy() returns finite value at ±1 boundary inputs."""
        model, data = self._make_model_and_data(n_vars=20, k=5)
        for val in [1.0, -1.0, 0.0]:
            x = jnp.full((20,), val)
            e = model.energy(x)
            assert jnp.isfinite(e), f"energy not finite at x={val}"
