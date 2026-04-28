"""Tests for Experiment 992: KAN MILP Violation Fix v2.

Verifies that enforce_monotonicity() correctly eliminates the 11 MILP violations
found in Exp 972 (7 monotonicity + 4 boundary/range violations).

Spec coverage: REQ-SAMPLE-015, REQ-KAN-VERIFY-001
"""

from __future__ import annotations

import os

import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from carnot.models.kaem_energy import KAEMEnergy, UnivariateKAEMLayer


# ---------------------------------------------------------------------------
# enforce_monotonicity unit tests
# ---------------------------------------------------------------------------


class TestEnforceMonotonicity:
    """REQ-KAN-VERIFY-001: enforce_monotonicity() must produce MILP-verified splines."""

    def _make_layer(self, ctrl_values: list[list[float]]) -> UnivariateKAEMLayer:
        """Helper: create a layer with known control points for testing."""
        n_vars = len(ctrl_values)
        n_knots = len(ctrl_values[0])
        key = jrandom.PRNGKey(0)
        layer = UnivariateKAEMLayer(n_vars=n_vars, n_knots=n_knots, key=key)
        # Override control points directly (bypass __init__ enforcement to test it)
        layer.control_points = jnp.array(ctrl_values, dtype=jnp.float32)
        return layer

    def test_non_decreasing_after_enforcement(self):
        """Control points must be non-decreasing after enforce_monotonicity().

        Spec: REQ-KAN-VERIFY-001 (monotonicity property must hold post-fix)
        """
        # Use known-violating control points (decreasing sequence)
        ctrl = [[0.1, 0.05, -0.1, 0.2, 0.15, -0.05, 0.3, 0.25]]
        layer = self._make_layer(ctrl)
        layer.enforce_monotonicity()

        ctrl_after = np.array(layer.control_points[0])
        diffs = np.diff(ctrl_after)
        assert np.all(diffs >= -1e-7), (
            f"Control points not non-decreasing after enforcement: diffs={diffs}"
        )

    def test_minimum_is_zero(self):
        """Minimum control point per variable must be 0.0 after enforcement.

        The zero-floor shift ensures energy(-1) = 0 per variable, fixing
        the boundary polarity violation seen in Exp 972.

        Spec: REQ-KAN-VERIFY-001 (boundary condition must hold)
        """
        ctrl = [[0.5, 0.3, 0.1, 0.8, 0.6, 0.2, 0.9, 0.7]]
        layer = self._make_layer(ctrl)
        layer.enforce_monotonicity()

        min_val = float(jnp.min(layer.control_points[0]))
        assert abs(min_val) < 1e-6, f"Minimum after enforcement is {min_val}, expected ~0"

    def test_maximum_bounded_by_one(self):
        """Maximum control point per variable must be <= 1.0 after enforcement.

        This satisfies the N-spins output range bound (total energy <= n_vars).

        Spec: REQ-KAN-VERIFY-001 (output range property must hold)
        """
        # Large values that would violate the output range bound
        ctrl = [[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]]
        layer = self._make_layer(ctrl)
        layer.enforce_monotonicity()

        max_val = float(jnp.max(layer.control_points[0]))
        assert max_val <= 1.0 + 1e-6, f"Max after enforcement is {max_val}, expected <= 1.0"

    def test_flat_spline_unchanged(self):
        """A flat (all-zero) spline must remain valid after enforcement.

        Avoids division-by-zero in the max-normalization step.

        Spec: REQ-SAMPLE-015 (degenerate model must not crash)
        """
        ctrl = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        layer = self._make_layer(ctrl)
        layer.enforce_monotonicity()  # must not raise

        ctrl_after = np.array(layer.control_points[0])
        assert np.all(np.isfinite(ctrl_after)), (
            "Flat spline contains non-finite values after enforcement"
        )

    def test_multi_variable_all_monotone(self):
        """All variables must be monotone after enforcement, not just the first.

        Spec: REQ-KAN-VERIFY-001 (applies to all n_vars variables)
        """
        # Four variables with different violation patterns
        ctrl = [
            [0.1, -0.1, 0.05, -0.05, 0.2, 0.1, -0.2, 0.3],  # many violations
            [0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4],  # strictly decreasing
            [0.0, 0.5, 0.3, 0.8, 0.6, 1.0, 0.9, 1.1],  # partial violations
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05],  # flat
        ]
        layer = self._make_layer(ctrl)
        layer.enforce_monotonicity()

        ctrl_after = np.array(layer.control_points)
        for i in range(4):
            diffs = np.diff(ctrl_after[i])
            assert np.all(diffs >= -1e-7), (
                f"Variable {i} not monotone after enforcement: diffs={diffs}"
            )
            assert ctrl_after[i].min() >= -1e-7, f"Variable {i} min={ctrl_after[i].min()} not >= 0"
            assert ctrl_after[i].max() <= 1.0 + 1e-6, (
                f"Variable {i} max={ctrl_after[i].max()} not <= 1.0"
            )


# ---------------------------------------------------------------------------
# Fresh model initialization tests
# ---------------------------------------------------------------------------


class TestFreshModelInit:
    """Fresh KAEMEnergy must pass MILP verification without any fit() call.

    Spec: REQ-KAN-VERIFY-001 (monotonicity must hold at init, not only post-fit)
    """

    def test_fresh_control_points_monotone(self):
        """Random init + enforce_monotonicity at __init__ must yield monotone splines."""
        for seed in [0, 1, 42, 99, 123]:
            key = jrandom.PRNGKey(seed)
            layer = UnivariateKAEMLayer(n_vars=4, n_knots=8, key=key)
            ctrl = np.array(layer.control_points)
            for i in range(layer.n_vars):
                diffs = np.diff(ctrl[i])
                assert np.all(diffs >= -1e-7), (
                    f"seed={seed}, var={i}: control points not monotone at init"
                )

    def test_fresh_boundary_condition(self):
        """energy(-1,...,-1) <= energy(+1,...,+1) for any freshly created model."""
        key = jrandom.PRNGKey(42)
        model = KAEMEnergy(n_vars=4, n_hidden=8, key=key)
        x_no_viol = jnp.full((4,), -1.0)
        x_max_viol = jnp.full((4,), 1.0)
        e_low = float(model.energy(x_no_viol))
        e_high = float(model.energy(x_max_viol))
        assert e_low <= e_high + 1e-6, (
            f"Boundary condition violated at init: energy(-1)={e_low} > energy(+1)={e_high}"
        )

    def test_fresh_output_range(self):
        """Total energy bounds stay within [-n_vars, n_vars] at init."""
        n_vars = 4
        key = jrandom.PRNGKey(42)
        model = KAEMEnergy(n_vars=n_vars, n_hidden=8, key=key)
        ctrl = np.array(model.layer.control_points)
        total_min = float(ctrl.min(axis=1).sum())
        total_max = float(ctrl.max(axis=1).sum())
        assert total_min >= -n_vars, f"total_min={total_min} < -{n_vars}"
        assert total_max <= n_vars, f"total_max={total_max} > {n_vars}"


# ---------------------------------------------------------------------------
# Fitted model verification
# ---------------------------------------------------------------------------


class TestFittedModelVerification:
    """KAEMEnergy.fit() must maintain MILP properties throughout training.

    Spec: REQ-SAMPLE-015, REQ-KAN-VERIFY-001
    """

    def test_fitted_monotonicity_held(self):
        """All variables must remain monotone after fit().

        enforce_monotonicity() is called at end of every epoch in fit().
        """
        key = jrandom.PRNGKey(42)
        model = KAEMEnergy(n_vars=4, n_hidden=8, key=key)

        rng = np.random.default_rng(42)
        data = jnp.array(
            rng.choice([-1.0, 0.0, 1.0], size=(200, 4), p=[0.2, 0.3, 0.5]).astype(np.float32)
        )
        model.fit(data, n_epochs=20)

        ctrl = np.array(model.layer.control_points)
        for i in range(4):
            diffs = np.diff(ctrl[i])
            assert np.all(diffs >= -1e-7), (
                f"var={i}: monotonicity violated after fit: diffs={diffs}"
            )

    def test_fitted_output_range_held(self):
        """Total energy must stay within [-n_vars, n_vars] after fit()."""
        n_vars = 4
        key = jrandom.PRNGKey(42)
        model = KAEMEnergy(n_vars=n_vars, n_hidden=8, key=key)

        rng = np.random.default_rng(42)
        data = jnp.array(
            rng.choice([-1.0, 0.0, 1.0], size=(200, n_vars), p=[0.2, 0.3, 0.5]).astype(np.float32)
        )
        model.fit(data, n_epochs=20)

        ctrl = np.array(model.layer.control_points)
        total_max = float(ctrl.max(axis=1).sum())
        assert total_max <= n_vars + 1e-6, (
            f"total_max={total_max:.4f} > {n_vars} (N spins bound violated)"
        )

    def test_fitted_boundary_condition_held(self):
        """energy(-1,...,-1) <= energy(+1,...,+1) must hold after fit()."""
        n_vars = 4
        key = jrandom.PRNGKey(42)
        model = KAEMEnergy(n_vars=n_vars, n_hidden=8, key=key)

        rng = np.random.default_rng(42)
        data = jnp.array(
            rng.choice([-1.0, 0.0, 1.0], size=(200, n_vars), p=[0.2, 0.3, 0.5]).astype(np.float32)
        )
        model.fit(data, n_epochs=20)

        x_no = jnp.full((n_vars,), -1.0)
        x_max = jnp.full((n_vars,), 1.0)
        e_no = float(model.energy(x_no))
        e_max = float(model.energy(x_max))
        assert e_no <= e_max + 1e-6, (
            f"Boundary violated after fit: energy(-1)={e_no:.4f} > energy(+1)={e_max:.4f}"
        )


# ---------------------------------------------------------------------------
# Result JSON schema test
# ---------------------------------------------------------------------------


class TestResultJSON:
    """The deliverable JSON must contain all required schema fields.

    Spec: experiment result schema (exp992_v1)
    """

    def test_result_json_exists_and_schema_valid(self):
        """results/experiment_992_kan_milp_violation_fix_v2.json must have required fields."""
        import json

        result_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "results",
            "experiment_992_kan_milp_violation_fix_v2.json",
        )
        assert os.path.exists(result_path), f"Result JSON not found at {result_path}"

        with open(result_path) as f:
            data = json.load(f)

        required_fields = [
            "violations_before",
            "violations_after",
            "monotonicity_violations_fixed",
            "boundary_violations_fixed",
            "speedup_ratio_after_fix",
            "kan_milp_verified",
            "honest_verdict",
        ]
        for field in required_fields:
            assert field in data, f"Required field '{field}' missing from result JSON"

        assert isinstance(data["violations_before"], int)
        assert isinstance(data["violations_after"], int)
        assert data["violations_before"] == 11
        assert data["violations_after"] == 0, (
            f"Expected 0 violations after fix, got {data['violations_after']}"
        )
        assert data["kan_milp_verified"] is True
        assert data["honest_verdict"] == "violations_fixed"
        assert data["speedup_ratio_after_fix"] > 0.0
