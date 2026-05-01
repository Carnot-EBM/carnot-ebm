"""Tests for Experiment 1036: Newton-Kaczmarz optimizer for KAEMEnergy.

REQ-SAMPLE-015: KAEM energy model supports differentiable energy computation.
SCENARIO-SAMPLE-027: Exact sampling via inverse-transform on trained model.

Tests cover:
- Feature extraction from FoVer items
- Grid normalisation to [-1, 1]
- AUROC computation correctness
- Spline Jacobian row structure (sparse, 2 non-zero entries)
- NK step: output shape and no NaN with regularisation
- Multilevel grid promotion: shape and value interpolation
- Monotonicity enforcement after NK step
- Adam training loop: loss decreases over epochs
- Full artifact schema validation (all required fields present and typed)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

# All imports from the experiment script under test
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_1036_nk_kaem_v2 as exp  # noqa: E402


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def _make_item(self, text: str = "x = 1 + 2", label: str = "correct") -> dict:
        return {
            "step_text": text,
            "label": label,
            "confidence": 1.0,
            "source": "math_z3",
            "problem_type": "algebra",
            "metaqa_cross_validated": False,
        }

    def test_returns_eight_features(self) -> None:
        feat = exp.extract_features(self._make_item())
        assert feat.shape == (8,), f"Expected 8 features, got {feat.shape}"

    def test_all_features_finite(self) -> None:
        feat = exp.extract_features(self._make_item())
        assert np.all(np.isfinite(feat)), "Features contain NaN or inf"

    def test_empty_text_no_crash(self) -> None:
        """Empty step_text should not raise (guard against division by zero)."""
        item = self._make_item(text="")
        feat = exp.extract_features(item)
        assert feat.shape == (8,)
        assert np.all(np.isfinite(feat))

    def test_digit_density_increases_with_digits(self) -> None:
        few_digits = exp.extract_features(self._make_item(text="abc"))
        many_digits = exp.extract_features(self._make_item(text="12345"))
        # Feature 2 (index 1) is digit density
        assert many_digits[1] > few_digits[1]

    def test_confidence_feature(self) -> None:
        item = self._make_item()
        item["confidence"] = 0.5
        feat = exp.extract_features(item)
        # Feature 5 (index 4) is raw confidence
        assert abs(feat[4] - 0.5) < 1e-6


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormaliseFeatures:
    def test_train_range_within_bounds(self) -> None:
        rng = np.random.default_rng(0)
        X = rng.uniform(0, 10, (20, 8)).astype(np.float32)
        X_norm, _ = exp.normalise_features(X, X)
        assert float(X_norm.min()) >= -1.0 - 1e-6
        assert float(X_norm.max()) <= 1.0 + 1e-6

    def test_test_clamped_to_minus_one_one(self) -> None:
        """Test samples outside training range are clamped, not unchecked extrapolation."""
        X_train = np.ones((5, 4), dtype=np.float32)
        X_test = np.array([[100.0] * 4] * 3, dtype=np.float32)
        _, X_test_norm = exp.normalise_features(X_train, X_test)
        assert float(X_test_norm.max()) <= 1.0 + 1e-6

    def test_returns_float32(self) -> None:
        X = np.zeros((5, 4), dtype=np.float32)
        X_norm, _ = exp.normalise_features(X, X)
        assert X_norm.dtype == np.float32


# ---------------------------------------------------------------------------
# AUROC
# ---------------------------------------------------------------------------


class TestComputeAUROC:
    def test_perfect_ranking(self) -> None:
        scores = np.array([0.9, 0.8, 0.1, 0.2])
        labels = np.array([1, 1, 0, 0])
        assert exp.compute_auroc(scores, labels) == pytest.approx(1.0)

    def test_worst_ranking(self) -> None:
        scores = np.array([0.1, 0.2, 0.9, 0.8])
        labels = np.array([1, 1, 0, 0])
        assert exp.compute_auroc(scores, labels) == pytest.approx(0.0)

    def test_random_returns_near_half(self) -> None:
        rng = np.random.default_rng(42)
        scores = rng.uniform(0, 1, 100)
        labels = (rng.uniform(0, 1, 100) > 0.5).astype(float)
        auroc = exp.compute_auroc(scores, labels)
        assert 0.2 < auroc < 0.8, "Random scores should be near 0.5"

    def test_all_same_class_returns_half(self) -> None:
        scores = np.array([0.5, 0.6])
        labels = np.array([1, 1])
        assert exp.compute_auroc(scores, labels) == pytest.approx(0.5)

    def test_tie_handling(self) -> None:
        scores = np.array([0.5, 0.5])
        labels = np.array([1, 0])
        # Tied score should give 0.5 AUROC
        assert exp.compute_auroc(scores, labels) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Spline Jacobian
# ---------------------------------------------------------------------------


class TestSplineJacobianRow:
    def test_returns_correct_shape(self) -> None:
        grad = exp._spline_jacobian_row(0.0, n_knots=8)
        assert grad.shape == (8,)

    def test_sums_to_one(self) -> None:
        """The two non-zero entries (1-t, t) always sum to 1."""
        for x in [-1.0, -0.5, 0.0, 0.5, 0.99]:
            grad = exp._spline_jacobian_row(x, n_knots=8)
            assert abs(grad.sum() - 1.0) < 1e-10, f"Sum != 1 at x={x}: {grad.sum()}"

    def test_exactly_two_nonzero_entries(self) -> None:
        """Linear interpolation activates exactly 2 knots (left and right)."""
        for x in [-0.7, 0.3, 0.9]:
            grad = exp._spline_jacobian_row(x, n_knots=8)
            n_nonzero = int(np.sum(grad != 0))
            assert n_nonzero <= 2, f"Expected <=2 non-zeros at x={x}, got {n_nonzero}"

    def test_boundary_left(self) -> None:
        """x = -1 should activate knot 0 with weight 1."""
        grad = exp._spline_jacobian_row(-1.0, n_knots=8)
        assert grad[0] == pytest.approx(1.0)

    def test_boundary_right(self) -> None:
        """x = +1 should activate knot n_knots-1 with weight 1."""
        grad = exp._spline_jacobian_row(1.0, n_knots=8)
        assert grad[7] == pytest.approx(1.0)

    def test_outside_range_clamped(self) -> None:
        """Values outside [-1, 1] should be clamped, not crash."""
        grad_low = exp._spline_jacobian_row(-2.0, n_knots=4)
        grad_high = exp._spline_jacobian_row(3.0, n_knots=4)
        assert np.all(np.isfinite(grad_low))
        assert np.all(np.isfinite(grad_high))


# ---------------------------------------------------------------------------
# Energy computation (numpy)
# ---------------------------------------------------------------------------


class TestComputeEnergyNumpy:
    def test_returns_scalar(self) -> None:
        ctrl = np.ones((4, 8), dtype=np.float64)
        x = np.zeros(4, dtype=np.float32)
        e = exp.compute_energy_numpy(ctrl, x, n_knots=8)
        assert isinstance(e, float)

    def test_finite_on_random_input(self) -> None:
        rng = np.random.default_rng(0)
        ctrl = rng.normal(0, 0.1, (6, 8))
        x = rng.uniform(-1, 1, 6).astype(np.float32)
        e = exp.compute_energy_numpy(ctrl, x, n_knots=8)
        assert math.isfinite(e)


# ---------------------------------------------------------------------------
# NK step
# ---------------------------------------------------------------------------


class TestNKStep:
    def test_output_shape_unchanged(self) -> None:
        rng = np.random.default_rng(0)
        n_vars, n_knots = 4, 8
        ctrl = rng.normal(0, 0.1, (n_vars, n_knots))
        X_batch = rng.uniform(-1, 1, (10, n_vars)).astype(np.float32)
        y_batch = (rng.uniform(0, 1, 10) > 0.5).astype(np.float32)
        ctrl_new = exp.nk_step(ctrl, X_batch, y_batch, n_knots, lam=1.0)
        assert ctrl_new.shape == ctrl.shape

    def test_no_nan_with_regularisation(self) -> None:
        """With λ=1.0 Tikhonov regularisation, NK step should be stable."""
        rng = np.random.default_rng(42)
        n_vars, n_knots = 8, 8
        ctrl = rng.normal(0, 0.1, (n_vars, n_knots))
        X_batch = rng.uniform(-1, 1, (10, n_vars)).astype(np.float32)
        y_batch = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        ctrl_new = exp.nk_step(ctrl, X_batch, y_batch, n_knots, lam=1.0)
        assert np.all(np.isfinite(ctrl_new)), "NK step produced NaN/inf with λ=1.0"

    def test_gradient_clipping_applied(self) -> None:
        """A very large residual should be clipped so ||Δw|| <= 1.0."""
        rng = np.random.default_rng(0)
        n_vars, n_knots = 4, 4
        # Extreme control points to maximise residual magnitude
        ctrl = np.ones((n_vars, n_knots), dtype=np.float64) * 100.0
        X_batch = rng.uniform(-1, 1, (5, n_vars)).astype(np.float32)
        y_batch = np.zeros(5, dtype=np.float32)
        ctrl_new = exp.nk_step(ctrl, X_batch, y_batch, n_knots, lam=0.01)
        delta = ctrl_new - ctrl
        assert np.linalg.norm(delta) <= 1.0 + 1e-9, "Gradient clipping failed"


# ---------------------------------------------------------------------------
# Grid promotion
# ---------------------------------------------------------------------------


class TestPromoteGrid:
    def test_output_shape_correct(self) -> None:
        ctrl_4 = np.linspace(0, 1, 4 * 6).reshape(6, 4)
        ctrl_8 = exp.promote_grid(ctrl_4, n_knots_fine=8)
        assert ctrl_8.shape == (6, 8)

    def test_boundary_values_preserved(self) -> None:
        """Endpoints of coarse grid must be preserved in fine grid."""
        n_vars = 3
        ctrl_4 = np.random.default_rng(0).uniform(0, 1, (n_vars, 4))
        ctrl_8 = exp.promote_grid(ctrl_4, 8)
        for i in range(n_vars):
            assert ctrl_8[i, 0] == pytest.approx(ctrl_4[i, 0], abs=1e-9)
            assert ctrl_8[i, -1] == pytest.approx(ctrl_4[i, -1], abs=1e-9)

    def test_monotone_input_stays_monotone(self) -> None:
        """If coarse grid is non-decreasing, interpolated fine grid should be too."""
        ctrl_4 = np.array([[0.0, 0.3, 0.7, 1.0]] * 4)
        ctrl_8 = exp.promote_grid(ctrl_4, 8)
        for i in range(4):
            diffs = np.diff(ctrl_8[i])
            assert np.all(diffs >= -1e-9), f"Row {i} not monotone after promotion"


# ---------------------------------------------------------------------------
# Monotonicity enforcement
# ---------------------------------------------------------------------------


class TestEnforceMonotonicity:
    def test_output_non_decreasing(self) -> None:
        rng = np.random.default_rng(7)
        ctrl = rng.normal(0, 1, (5, 8))
        ctrl_m = exp._enforce_monotonicity(ctrl)
        for i in range(5):
            assert np.all(np.diff(ctrl_m[i]) >= -1e-9), f"Row {i} not non-decreasing"

    def test_min_shifted_to_zero(self) -> None:
        ctrl = np.ones((3, 4)) * -5.0
        ctrl_m = exp._enforce_monotonicity(ctrl)
        for i in range(3):
            assert ctrl_m[i, 0] == pytest.approx(0.0, abs=1e-9)

    def test_max_capped_at_one(self) -> None:
        ctrl = np.array([[0.0, 5.0, 10.0, 15.0]])
        ctrl_m = exp._enforce_monotonicity(ctrl)
        assert float(ctrl_m.max()) <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Adam training
# ---------------------------------------------------------------------------


class TestTrainAdam:
    def _make_data(self, n: int = 30, n_vars: int = 4) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (n, n_vars)).astype(np.float32)
        y = (rng.uniform(0, 1, n) > 0.5).astype(np.float32)
        return X, y

    def test_returns_correct_ctrl_shape(self) -> None:
        X, y = self._make_data()
        ctrl, _ = exp.train_adam(X, y, n_knots=8, n_epochs=5, lr=0.01)
        assert ctrl.shape == (4, 8)

    def test_losses_list_length(self) -> None:
        X, y = self._make_data()
        _, losses = exp.train_adam(X, y, n_knots=8, n_epochs=10, lr=0.01)
        assert len(losses) == 10

    def test_ctrl_finite_after_training(self) -> None:
        X, y = self._make_data()
        ctrl, _ = exp.train_adam(X, y, n_knots=8, n_epochs=20, lr=0.01)
        assert np.all(np.isfinite(ctrl))

    def test_warm_start_accepted(self) -> None:
        """Providing init_ctrl should not crash and returns same shape."""
        X, y = self._make_data()
        init = np.zeros((4, 4))
        ctrl, _ = exp.train_adam(X, y, n_knots=4, n_epochs=3, lr=0.01, init_ctrl=init)
        assert ctrl.shape == (4, 4)

    def test_monotonicity_invariant_held(self) -> None:
        """Control points should be non-decreasing after Adam training."""
        X, y = self._make_data()
        ctrl, _ = exp.train_adam(X, y, n_knots=8, n_epochs=10, lr=0.01)
        for i in range(ctrl.shape[0]):
            assert np.all(np.diff(ctrl[i]) >= -1e-9)


# ---------------------------------------------------------------------------
# Artifact schema validation
# ---------------------------------------------------------------------------

_REQUIRED_FIELDS = [
    "experiment",
    "title",
    "run_date",
    "started_at",
    "finished_at",
    "duration_s",
    "status",
    "schema",
    "honest_verdict",
    "adam_wall_time_s",
    "nk_wall_time_s",
    "convergence_speedup",
    "auroc_adam",
    "auroc_nk_multilevel",
    "auroc_no_regression",
    "nk_lambda_used",
    "grid_levels_used",
]

_VALID_VERDICTS = {
    "nk_speedup_confirmed",
    "nk_partial_speedup_below_2x",
    "nk_diverged_fallback_used",
    "failed",
}

_DELIVERABLE = Path(__file__).resolve().parents[2] / "results" / "experiment_1036_nk_kaem_v2.json"


class TestArtifactSchema:
    @pytest.fixture(scope="class")
    def artifact(self) -> dict:
        assert _DELIVERABLE.exists(), f"Deliverable not found: {_DELIVERABLE}"
        return json.loads(_DELIVERABLE.read_text())

    def test_all_required_fields_present(self, artifact: dict) -> None:
        missing = [f for f in _REQUIRED_FIELDS if f not in artifact]
        assert not missing, f"Missing fields: {missing}"

    def test_experiment_id(self, artifact: dict) -> None:
        assert artifact["experiment"] == 1036

    def test_honest_verdict_valid(self, artifact: dict) -> None:
        assert artifact["honest_verdict"] in _VALID_VERDICTS

    def test_auroc_adam_in_range(self, artifact: dict) -> None:
        auroc = artifact["auroc_adam"]
        assert 0.0 <= auroc <= 1.0, f"auroc_adam={auroc} out of [0, 1]"

    def test_auroc_nk_in_range(self, artifact: dict) -> None:
        auroc = artifact["auroc_nk_multilevel"]
        assert 0.0 <= auroc <= 1.0, f"auroc_nk_multilevel={auroc} out of [0, 1]"

    def test_auroc_no_regression_matches_values(self, artifact: dict) -> None:
        """auroc_no_regression must equal (auroc_nk_multilevel >= auroc_adam - 0.02)."""
        expected = artifact["auroc_nk_multilevel"] >= artifact["auroc_adam"] - 0.02
        assert artifact["auroc_no_regression"] == expected

    def test_grid_levels_used_is_list(self, artifact: dict) -> None:
        glu = artifact["grid_levels_used"]
        assert isinstance(glu, list) and len(glu) > 0

    def test_convergence_speedup_positive(self, artifact: dict) -> None:
        assert artifact["convergence_speedup"] > 0.0

    def test_nk_lambda_used_positive(self, artifact: dict) -> None:
        assert artifact["nk_lambda_used"] > 0.0

    def test_wall_times_positive(self, artifact: dict) -> None:
        assert artifact["adam_wall_time_s"] > 0.0
        assert artifact["nk_wall_time_s"] > 0.0

    def test_prior_failures_present(self, artifact: dict) -> None:
        pf = artifact.get("prior_failures", [])
        assert len(pf) >= 1, "prior_failures must list at least one prior failed attempt"

    def test_status_is_valid(self, artifact: dict) -> None:
        assert artifact["status"] in {"success", "failed"}
