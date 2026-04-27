"""Tests for experiment_972 KAN MILP formal verification helpers.

Covers pwa_segments, verify_monotonicity, verify_output_range,
verify_boundary_condition, and verify_monotonicity_milp.

Spec: REQ-KAN-VERIFY-001
Scenario: SCENARIO-KAN-VERIFY-001
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

from experiment_972_kan_milp_formal_verification import (
    pwa_segments,
    verify_boundary_condition,
    verify_monotonicity,
    verify_monotonicity_milp,
    verify_output_range,
)
from carnot.models.kaem_energy import UnivariateKAEMLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def flat_layer() -> UnivariateKAEMLayer:
    """UnivariateKAEMLayer with all control points = 0 (flat energy landscape)."""
    import jax.random as jrandom

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jrandom.PRNGKey(0))
    import jax.numpy as jnp

    layer.control_points = jnp.zeros((2, 4))
    return layer


@pytest.fixture
def monotone_layer() -> UnivariateKAEMLayer:
    """UnivariateKAEMLayer with strictly increasing control points."""
    import jax.numpy as jnp
    import jax.random as jrandom

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jrandom.PRNGKey(0))
    # ctrl = [0.0, 1.0, 2.0, 3.0] — strictly increasing for both vars
    layer.control_points = jnp.array([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    return layer


@pytest.fixture
def non_monotone_layer() -> UnivariateKAEMLayer:
    """UnivariateKAEMLayer with a downward dip in control points."""
    import jax.numpy as jnp
    import jax.random as jrandom

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jrandom.PRNGKey(0))
    # ctrl[1] < ctrl[0] — clear monotonicity violation at knot 0->1
    layer.control_points = jnp.array([[3.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    return layer


# ---------------------------------------------------------------------------
# pwa_segments
# ---------------------------------------------------------------------------


def test_pwa_segments_count():
    """PWA segments count = n_knots - 1. Spec: REQ-KAN-VERIFY-001"""
    ctrl = np.array([0.0, 1.0, 2.0, 3.0])
    knots = np.linspace(-1.0, 1.0, 4)
    segs = pwa_segments(ctrl, knots)
    assert len(segs) == 3


def test_pwa_segments_slope_intercept():
    """Each segment's slope and intercept correctly interpolate between knot values.
    Spec: REQ-KAN-VERIFY-001"""
    ctrl = np.array([0.0, 2.0])
    knots = np.array([-1.0, 1.0])
    segs = pwa_segments(ctrl, knots)
    assert len(segs) == 1
    seg = segs[0]
    # slope = (2-0)/(1-(-1)) = 1.0; intercept = 0 - 1*(-1) = 1.0
    assert abs(seg["slope"] - 1.0) < 1e-6
    assert abs(seg["intercept"] - 1.0) < 1e-6
    # Verify at midpoint x=0: slope*0 + intercept = 1.0; ctrl interp = 1.0
    assert abs(seg["slope"] * 0.0 + seg["intercept"] - 1.0) < 1e-6


def test_pwa_segments_flat():
    """Flat spline produces zero-slope segments. Spec: REQ-KAN-VERIFY-001"""
    ctrl = np.array([5.0, 5.0, 5.0])
    knots = np.linspace(-1.0, 1.0, 3)
    segs = pwa_segments(ctrl, knots)
    for seg in segs:
        assert abs(seg["slope"]) < 1e-9
        assert abs(seg["intercept"] - 5.0) < 1e-6


# ---------------------------------------------------------------------------
# verify_monotonicity
# ---------------------------------------------------------------------------


def test_monotonicity_flat_layer_verified(flat_layer):
    """Flat energy (all ctrl=0) satisfies monotonicity (no decrease). Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity(flat_layer, var_idx=0)
    assert result["verified"] is True
    assert len(result["violation_detail"]) == 0


def test_monotonicity_increasing_verified(monotone_layer):
    """Strictly increasing ctrl points satisfies monotonicity. Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity(monotone_layer, var_idx=0)
    assert result["verified"] is True


def test_monotonicity_non_monotone_fails(non_monotone_layer):
    """Ctrl with downward dip reports monotonicity violation. Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity(non_monotone_layer, var_idx=0)
    assert result["verified"] is False
    assert len(result["violation_detail"]) > 0


def test_monotonicity_returns_energy_at_knots(monotone_layer):
    """Result contains energy_at_knots list with correct length. Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity(monotone_layer, var_idx=0)
    assert "energy_at_knots" in result
    assert len(result["energy_at_knots"]) == 4  # n_knots=4


# ---------------------------------------------------------------------------
# verify_output_range
# ---------------------------------------------------------------------------


def test_output_range_flat_verified(flat_layer):
    """All-zero control points stay within any positive N-spin bound.
    Spec: REQ-KAN-VERIFY-001"""
    result = verify_output_range(flat_layer, n_spins=2)
    assert result["verified"] is True
    assert result["total_min"] == 0.0
    assert result["total_max"] == 0.0


def test_output_range_large_ctrl_violated():
    """Control points exceeding N-spin bound trigger output range violation.
    Spec: REQ-KAN-VERIFY-001"""
    import jax.numpy as jnp
    import jax.random as jrandom

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jrandom.PRNGKey(0))
    # ctrl values of 100 far exceed n_spins=2 bound
    layer.control_points = jnp.full((2, 4), 100.0)
    result = verify_output_range(layer, n_spins=2)
    assert result["verified"] is False
    assert result["total_max"] > 2


def test_output_range_result_keys(flat_layer):
    """Result dict contains all expected keys. Spec: REQ-KAN-VERIFY-001"""
    result = verify_output_range(flat_layer, n_spins=4)
    for key in (
        "property",
        "verified",
        "violation_detail",
        "total_min",
        "total_max",
        "n_spins_bound",
    ):
        assert key in result


# ---------------------------------------------------------------------------
# verify_boundary_condition
# ---------------------------------------------------------------------------


def test_boundary_condition_monotone_verified(monotone_layer):
    """Monotone increasing layer: energy at x=-1 < x=+1. Spec: REQ-KAN-VERIFY-001"""
    result = verify_boundary_condition(monotone_layer)
    # ctrl=[0,1,2,3]: energy(-1) = ctrl[0]=0; energy(+1) = ctrl[3]=3 -> 0 < 3
    assert result["verified"] is True


def test_boundary_condition_inverted_violated():
    """Decreasing ctrl: energy at x=-1 > energy at x=+1 is a boundary violation.
    Spec: REQ-KAN-VERIFY-001"""
    import jax.numpy as jnp
    import jax.random as jrandom

    layer = UnivariateKAEMLayer(n_vars=2, n_knots=4, key=jrandom.PRNGKey(0))
    # Decreasing: high energy at x=-1, low at x=+1 (wrong polarity for violation scoring)
    layer.control_points = jnp.array([[3.0, 2.0, 1.0, 0.0], [3.0, 2.0, 1.0, 0.0]])
    result = verify_boundary_condition(layer)
    assert result["verified"] is False
    assert len(result["violation_detail"]) > 0


def test_boundary_condition_flat_verified(flat_layer):
    """Flat energy: e_no_violation == e_max_violation. Not a violation. Spec: REQ-KAN-VERIFY-001"""
    result = verify_boundary_condition(flat_layer)
    # Both are 0; not violated (0 not > 0 + 1e-6)
    assert result["verified"] is True


# ---------------------------------------------------------------------------
# verify_monotonicity_milp
# ---------------------------------------------------------------------------


def test_milp_monotonicity_flat_verified(flat_layer):
    """MILP finds no counter-example for flat spline. Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity_milp(flat_layer, var_idx=0)
    assert result["verified"] is True
    assert result["milp_obj"] <= 1e-5


def test_milp_monotonicity_increasing_verified(monotone_layer):
    """MILP finds no counter-example for strictly increasing spline. Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity_milp(monotone_layer, var_idx=0)
    assert result["verified"] is True


def test_milp_monotonicity_decreasing_violated():
    """MILP finds counter-example for decreasing spline. Spec: REQ-KAN-VERIFY-001"""
    import jax.numpy as jnp
    import jax.random as jrandom

    layer = UnivariateKAEMLayer(n_vars=1, n_knots=4, key=jrandom.PRNGKey(0))
    # Decreasing ctrl: energy(x=-1)=3, energy(x=+1)=0 — counter-example trivially exists
    layer.control_points = jnp.array([[3.0, 2.0, 1.0, 0.0]])
    result = verify_monotonicity_milp(layer, var_idx=0)
    assert result["verified"] is False
    assert result["milp_obj"] > 1e-6
    assert len(result["violation_detail"]) > 0


def test_milp_result_keys(flat_layer):
    """MILP result contains all expected keys. Spec: REQ-KAN-VERIFY-001"""
    result = verify_monotonicity_milp(flat_layer, var_idx=0)
    for key in ("property", "verified", "violation_detail", "milp_status", "milp_obj"):
        assert key in result
