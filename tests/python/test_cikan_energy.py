"""Tests for carnot.models.cikan_energy — CIKANEnergy, CIKANLayer, ConstraintBoundary.

100% coverage target for cikan_energy.py.

Spec: REQ-SAMPLE-025, REQ-SAMPLE-026,
      SCENARIO-SAMPLE-038, SCENARIO-SAMPLE-039, SCENARIO-SAMPLE-040
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

from carnot.models.cikan_energy import (
    CIKANEnergy,
    CIKANLayer,
    ConstraintBoundary,
)
from carnot.models.kaem_energy import UnivariateKAEMLayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _layer(n_vars: int = 2, n_knots_base: int = 8, boundary_k: int = 4) -> CIKANLayer:
    return CIKANLayer(n_vars=n_vars, n_knots_base=n_knots_base, boundary_k=boundary_k,
                      key=jrandom.PRNGKey(0))


def _model(n_vars: int = 2, n_hidden: int = 8) -> CIKANEnergy:
    return CIKANEnergy(n_vars=n_vars, n_hidden=n_hidden, key=jrandom.PRNGKey(0))


# ---------------------------------------------------------------------------
# ConstraintBoundary
# ---------------------------------------------------------------------------


class TestConstraintBoundary:
    """REQ-SAMPLE-025: ConstraintBoundary dataclass behaves correctly."""

    def test_defaults(self) -> None:
        """Default sharpness is 1.0."""
        b = ConstraintBoundary(position=0.5)
        assert b.position == pytest.approx(0.5)
        assert b.sharpness == pytest.approx(1.0)

    def test_custom_sharpness(self) -> None:
        """Custom sharpness is preserved."""
        b = ConstraintBoundary(position=-0.3, sharpness=2.5)
        assert b.position == pytest.approx(-0.3)
        assert b.sharpness == pytest.approx(2.5)

    def test_equality(self) -> None:
        """Dataclass equality works."""
        b1 = ConstraintBoundary(0.0, 1.0)
        b2 = ConstraintBoundary(0.0, 1.0)
        assert b1 == b2


# ---------------------------------------------------------------------------
# CIKANLayer — initialisation
# ---------------------------------------------------------------------------


class TestCIKANLayerInit:
    """REQ-SAMPLE-025: CIKANLayer initialises correctly."""

    def test_inherits_from_univariate(self) -> None:
        """CIKANLayer is a subclass of UnivariateKAEMLayer."""
        layer = _layer()
        assert isinstance(layer, UnivariateKAEMLayer)

    def test_base_knot_count(self) -> None:
        """Initial knot count matches n_knots_base."""
        layer = _layer(n_knots_base=6)
        assert layer.n_knots_base == 6
        assert layer.n_knots == 6

    def test_boundary_k_stored(self) -> None:
        """boundary_k attribute is stored."""
        layer = _layer(boundary_k=3)
        assert layer.boundary_k == 3

    def test_control_points_shape(self) -> None:
        """Control points shape is (n_vars, n_knots_base) at init."""
        layer = _layer(n_vars=3, n_knots_base=7)
        assert layer.control_points.shape == (3, 7)

    def test_default_key(self) -> None:
        """Layer initialises without a key."""
        layer = CIKANLayer(n_vars=2)
        assert layer.n_vars == 2


# ---------------------------------------------------------------------------
# CIKANLayer — _distribute_knots_with_boundaries
# ---------------------------------------------------------------------------


class TestDistributeKnotsWithBoundaries:
    """REQ-SAMPLE-025, SCENARIO-SAMPLE-038: extra knots near boundaries."""

    def test_no_boundaries_returns_base_grid(self) -> None:
        """With empty boundaries list, returns the uniform base grid."""
        layer = _layer(n_knots_base=8)
        knots = layer._distribute_knots_with_boundaries([], data_std=0.3)
        assert len(knots) == 8
        assert float(knots[0]) == pytest.approx(-1.0)
        assert float(knots[-1]) == pytest.approx(1.0)

    def test_boundary_adds_extra_knots(self) -> None:
        """With a boundary, result has more knots than n_knots_base."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        boundaries = [ConstraintBoundary(0.5, sharpness=1.0)]
        knots = layer._distribute_knots_with_boundaries(boundaries, data_std=0.3)
        # boundary_k=4 means 3 extra knots → total > 8
        assert len(knots) > 8

    def test_knots_sorted(self) -> None:
        """Returned knots are in ascending order."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        boundaries = [ConstraintBoundary(0.0)]
        knots = layer._distribute_knots_with_boundaries(boundaries, data_std=0.3)
        assert list(knots) == sorted(knots)

    def test_knots_in_domain(self) -> None:
        """All knot positions are within [-1, 1]."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        boundaries = [ConstraintBoundary(0.9, sharpness=2.0)]
        knots = layer._distribute_knots_with_boundaries(boundaries, data_std=0.5)
        assert np.all(knots >= -1.0)
        assert np.all(knots <= 1.0)

    def test_multiple_boundaries(self) -> None:
        """Multiple boundaries each add extra knots."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        boundaries = [ConstraintBoundary(-0.5), ConstraintBoundary(0.5)]
        knots_two = layer._distribute_knots_with_boundaries(boundaries, data_std=0.3)
        knots_zero = layer._distribute_knots_with_boundaries([], data_std=0.3)
        assert len(knots_two) > len(knots_zero)

    def test_boundary_k1_no_extra(self) -> None:
        """boundary_k=1 means 0 extra knots (k-1=0), returns base grid."""
        layer = CIKANLayer(n_vars=2, n_knots_base=8, boundary_k=1)
        boundaries = [ConstraintBoundary(0.0)]
        knots = layer._distribute_knots_with_boundaries(boundaries, data_std=0.3)
        # No extras: should equal base grid (deduplicated)
        assert len(knots) == 8

    def test_cikan_has_more_knots_near_boundary_than_uniform(self) -> None:
        """SCENARIO-SAMPLE-038: CIKANLayer with boundary has more knots than uniform."""
        # SCENARIO-SAMPLE-038 core assertion
        uniform_layer = UnivariateKAEMLayer(n_vars=1, n_knots=8)
        cikan_layer = CIKANLayer(n_vars=1, n_knots_base=8, boundary_k=4)
        boundaries = [ConstraintBoundary(0.5)]
        cikan_knots = cikan_layer._distribute_knots_with_boundaries(boundaries, data_std=0.3)
        assert len(cikan_knots) > uniform_layer.n_knots

    def test_window_clipped_at_boundary_of_domain(self) -> None:
        """Boundary at -0.95 with large sharpness does not produce knots < -1."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        boundaries = [ConstraintBoundary(-0.95, sharpness=5.0)]
        knots = layer._distribute_knots_with_boundaries(boundaries, data_std=0.5)
        assert np.all(knots >= -1.0)

    def test_degenerate_window_hi_equals_lo(self) -> None:
        """Boundary at domain edge with zero-width window still returns base grid."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        # sharpness=0 → half_width=0 → lo==hi → no extra knots inserted
        boundaries = [ConstraintBoundary(-1.0, sharpness=0.0)]
        knots = layer._distribute_knots_with_boundaries(boundaries, data_std=0.3)
        assert len(knots) == 8


# ---------------------------------------------------------------------------
# CIKANLayer — apply_boundaries
# ---------------------------------------------------------------------------


class TestApplyBoundaries:
    """REQ-SAMPLE-025: apply_boundaries updates layer state correctly."""

    def test_knot_count_grows(self) -> None:
        """After apply_boundaries, n_knots > n_knots_base."""
        layer = _layer(n_knots_base=8, boundary_k=4)
        layer.apply_boundaries([ConstraintBoundary(0.0)], data_std=0.3)
        assert layer.n_knots > layer.n_knots_base

    def test_control_points_updated(self) -> None:
        """control_points shape matches new n_knots after apply_boundaries."""
        layer = _layer(n_vars=3, n_knots_base=8, boundary_k=4)
        layer.apply_boundaries([ConstraintBoundary(0.0)], data_std=0.3)
        assert layer.control_points.shape == (3, layer.n_knots)

    def test_knots_jax_array(self) -> None:
        """After apply_boundaries, _knots is a JAX array."""
        layer = _layer()
        layer.apply_boundaries([ConstraintBoundary(0.0)], data_std=0.3)
        assert isinstance(layer._knots, jax.Array)


# ---------------------------------------------------------------------------
# CIKANEnergy — initialisation
# ---------------------------------------------------------------------------


class TestCIKANEnergyInit:
    """REQ-SAMPLE-025: CIKANEnergy initialises correctly."""

    def test_inherits_from_kaem_energy(self) -> None:
        """CIKANEnergy is a subclass of KAEMEnergy."""
        from carnot.models.kaem_energy import KAEMEnergy
        model = _model()
        assert isinstance(model, KAEMEnergy)

    def test_layer_is_cikan_layer(self) -> None:
        """self.layer is a CIKANLayer after init."""
        model = _model()
        assert isinstance(model.layer, CIKANLayer)

    def test_boundaries_default_empty(self) -> None:
        """Default boundaries is an empty list."""
        model = _model()
        assert model.boundaries == []

    def test_boundaries_passed_at_init(self) -> None:
        """Boundaries passed at init are stored."""
        b = ConstraintBoundary(0.5)
        model = CIKANEnergy(n_vars=2, n_hidden=8, boundaries=[b])
        assert len(model.boundaries) == 1
        assert model.boundaries[0] == b

    def test_default_key(self) -> None:
        """CIKANEnergy can be constructed without a key."""
        model = CIKANEnergy(n_vars=2)
        assert model.n_vars == 2


# ---------------------------------------------------------------------------
# CIKANEnergy — energy (differentiability)
# ---------------------------------------------------------------------------


class TestCIKANEnergyDifferentiability:
    """SCENARIO-SAMPLE-039: CIKANEnergy.energy() is JAX-differentiable."""

    def test_energy_is_scalar(self) -> None:
        """energy(x) returns a scalar."""
        model = _model(n_vars=3)
        x = jnp.zeros(3)
        e = model.energy(x)
        assert e.shape == ()

    def test_energy_grad_finite(self) -> None:
        """jax.grad(model.energy)(x) is finite for all x in [-1,1]^n."""
        model = _model(n_vars=3)
        x = jnp.array([0.1, -0.5, 0.9])
        grad = jax.grad(model.energy)(x)
        assert grad.shape == (3,)
        assert jnp.all(jnp.isfinite(grad))

    def test_energy_grad_at_boundary(self) -> None:
        """Gradient is finite at the boundary position."""
        model = _model(n_vars=1)
        x = jnp.array([0.0])
        grad = jax.grad(model.energy)(x)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# CIKANEnergy — fit_with_constraints
# ---------------------------------------------------------------------------


class TestFitWithConstraints:
    """SCENARIO-SAMPLE-040: fit_with_constraints completes and sets state."""

    def test_completes_without_error(self) -> None:
        """fit_with_constraints runs without raising for simple 1D data."""
        model = CIKANEnergy(n_vars=1, n_hidden=8)
        data = jnp.array(np.random.default_rng(0).uniform(0.0, 1.0, (50, 1)).astype(np.float32))
        boundaries = [ConstraintBoundary(0.0)]
        losses = model.fit_with_constraints(data, boundaries, n_epochs=5)
        assert isinstance(losses, list)
        assert len(losses) == 5

    def test_boundaries_stored(self) -> None:
        """self.boundaries is set to the provided list after fit_with_constraints."""
        model = _model(n_vars=1)
        data = jnp.ones((20, 1)) * 0.5
        boundaries = [ConstraintBoundary(0.0), ConstraintBoundary(0.5)]
        model.fit_with_constraints(data, boundaries, n_epochs=2)
        assert len(model.boundaries) == 2

    def test_knot_count_increases(self) -> None:
        """After fit_with_constraints, layer has more knots than base."""
        model = CIKANEnergy(n_vars=1, n_hidden=8)
        data = jnp.array(np.linspace(-1, 1, 40).reshape(40, 1).astype(np.float32))
        model.fit_with_constraints(data, [ConstraintBoundary(0.0)], n_epochs=2)
        assert model.layer.n_knots > 8

    def test_trivial_data_std_fallback(self) -> None:
        """If data std < 1e-6, data_std falls back to 0.3 without error."""
        model = CIKANEnergy(n_vars=1, n_hidden=8)
        # All-same data → std = 0
        data = jnp.zeros((20, 1))
        boundaries = [ConstraintBoundary(0.0)]
        losses = model.fit_with_constraints(data, boundaries, n_epochs=2)
        assert isinstance(losses, list)

    def test_energy_still_differentiable_after_fit(self) -> None:
        """energy() remains differentiable after fit_with_constraints."""
        model = CIKANEnergy(n_vars=1, n_hidden=8)
        data = jnp.array(np.random.default_rng(1).uniform(-1, 1, (20, 1)).astype(np.float32))
        model.fit_with_constraints(data, [ConstraintBoundary(0.0)], n_epochs=2)
        x = jnp.array([0.1])
        grad = jax.grad(model.energy)(x)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# Export check
# ---------------------------------------------------------------------------


def test_exports_from_models_init() -> None:
    """CIKANEnergy, CIKANLayer, ConstraintBoundary are exported from carnot.models."""
    from carnot.models import CIKANEnergy as CE, CIKANLayer as CL, ConstraintBoundary as CB
    assert CE is CIKANEnergy
    assert CL is CIKANLayer
    assert CB is ConstraintBoundary
