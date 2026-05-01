"""Tests for Experiment 1045: Probe Ensemble v5 — ThinkPRM + GS-KAN + NK-KAEM.

REQ-VERIFY-098: ThinkPRM probe outputs AUROC on FoVer binary classification.
REQ-LEARN-011: Probe training converges without NaN on FoVer corpus.
REQ-SAMPLE-015: Energy model interface: energy() returns scalar; fit() trains model.
SCENARIO-VERIFY-130: Trained probe AUROC exceeds random baseline on held-out test split.

Tests cover:
- extract_text_features: shape, finiteness, empty-text safety.
- load_split: feature matrix shape and label conventions.
- normalise_features: output range, test-set clamping.
- compute_auroc: perfect discrimination, random baseline, inverted scores.
- LogisticProbe: convergence on linearly-separable data (1D and 8D).
- LogisticProbe: predict_proba output range [0, 1].
- _spline_jac_row: shape, sparsity (exactly 2 non-zeros), sums-to-one.
- _eval_energy: monotone output for monotone ctrl.
- _enforce_mono: non-decreasing + min-zero + max-one invariants.
- _promote_grid: output shape, boundary value preservation.
- _nk_step: output shape, no NaN with λ > 0.
- Artifact: required schema fields all present and correctly typed.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import experiment_1045_probe_ensemble_v5 as exp  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(text: str = "x = 1 + 2", label: str = "correct") -> dict:
    return {
        "step_text": text,
        "label": label,
        "confidence": 1.0,
        "source": "math_z3",
        "problem_type": "algebra",
    }


def _small_xy(n: int = 20, n_vars: int = 8, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return synthetic X (n, n_vars) in [-1,1] and binary y."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.0, 1.0, (n, n_vars)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.float32)
    return X, y


# ---------------------------------------------------------------------------
# extract_text_features
# ---------------------------------------------------------------------------


class TestExtractTextFeatures:
    def test_shape(self) -> None:
        feat = exp.extract_text_features(_make_item())
        assert feat.shape == (8,)

    def test_all_finite(self) -> None:
        feat = exp.extract_text_features(_make_item())
        assert np.all(np.isfinite(feat))

    def test_empty_text_no_crash(self) -> None:
        """Empty step_text must not raise (guards division by zero on n_chars)."""
        feat = exp.extract_text_features(_make_item(text=""))
        assert feat.shape == (8,)
        assert np.all(np.isfinite(feat))

    def test_confidence_feature_reflects_item(self) -> None:
        """Feature 4 (index 4) should equal the confidence value."""
        item = _make_item()
        item["confidence"] = 0.75
        feat = exp.extract_text_features(item)
        assert abs(float(feat[4]) - 0.75) < 1e-5

    def test_source_feature_in_range(self) -> None:
        """Source categorical feature (index 5) must be in [0, 1]."""
        for src in ["math_z3", "fover", "other", "unknown_src"]:
            item = _make_item()
            item["source"] = src
            feat = exp.extract_text_features(item)
            assert 0.0 <= float(feat[5]) <= 1.0, f"src={src} feature out of range"


# ---------------------------------------------------------------------------
# normalise_features
# ---------------------------------------------------------------------------


class TestNormaliseFeatures:
    def test_train_in_minus1_1(self) -> None:
        rng = np.random.default_rng(1)
        X_train = rng.uniform(0, 10, (30, 8)).astype(np.float32)
        X_test = rng.uniform(0, 10, (10, 8)).astype(np.float32)
        X_tn, X_te = exp.normalise_features(X_train, X_test)
        assert float(X_tn.min()) >= -1.0 - 1e-5
        assert float(X_tn.max()) <= 1.0 + 1e-5

    def test_test_clipped(self) -> None:
        """Test set values outside training range must be clipped to [-1, 1]."""
        X_train = np.array([[0.0] * 8, [1.0] * 8], dtype=np.float32)
        X_test = np.array([[5.0] * 8], dtype=np.float32)  # out of training range
        _, X_te = exp.normalise_features(X_train, X_test)
        assert float(X_te.max()) <= 1.0 + 1e-5

    def test_output_dtype_float32(self) -> None:
        X_train = np.ones((10, 8), dtype=np.float32)
        X_test = np.ones((5, 8), dtype=np.float32)
        X_tn, X_te = exp.normalise_features(X_train, X_test)
        assert X_tn.dtype == np.float32
        assert X_te.dtype == np.float32


# ---------------------------------------------------------------------------
# compute_auroc
# ---------------------------------------------------------------------------


class TestComputeAuroc:
    def test_perfect_discrimination(self) -> None:
        """Positives always score higher than negatives → AUROC=1.0."""
        scores = np.array([0.9, 0.8, 0.1, 0.05])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert abs(exp.compute_auroc(scores, labels) - 1.0) < 1e-6

    def test_inverted_scores(self) -> None:
        """Positives always score lower than negatives → AUROC=0.0."""
        scores = np.array([0.1, 0.05, 0.9, 0.8])
        labels = np.array([1.0, 1.0, 0.0, 0.0])
        assert abs(exp.compute_auroc(scores, labels) - 0.0) < 1e-6

    def test_random_baseline(self) -> None:
        """Tied scores → AUROC=0.5."""
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([1.0, 0.0, 1.0, 0.0])
        assert abs(exp.compute_auroc(scores, labels) - 0.5) < 1e-6

    def test_degenerate_single_class(self) -> None:
        """Only one class in labels → returns 0.5 (undefined AUROC)."""
        scores = np.array([0.8, 0.6, 0.4])
        labels = np.array([1.0, 1.0, 1.0])
        assert exp.compute_auroc(scores, labels) == 0.5


# ---------------------------------------------------------------------------
# LogisticProbe
# ---------------------------------------------------------------------------


class TestLogisticProbe:
    def test_1d_converges_on_separable(self) -> None:
        """1D probe should achieve training AUROC > 0.8 on linearly separable data."""
        rng = np.random.default_rng(0)
        # Positives at x=1.0 + noise, negatives at x=-1.0 + noise
        x_pos = rng.normal(1.0, 0.1, 20)
        x_neg = rng.normal(-1.0, 0.1, 20)
        X = np.concatenate([x_pos, x_neg]).reshape(-1, 1)
        y = np.array([1.0] * 20 + [0.0] * 20)

        probe = exp.LogisticProbe(n_features=1, lr=0.5, n_epochs=300, reg=0.001)
        probe.train(X, y)

        preds = probe.predict_proba(X)
        auroc = exp.compute_auroc(preds, y)
        assert auroc > 0.8, f"Expected AUROC > 0.8 on separable 1D data, got {auroc:.4f}"

    def test_8d_no_nan(self) -> None:
        """8D probe training must not produce NaN weights or predictions."""
        X, y = _small_xy(n=40, n_vars=8)
        probe = exp.LogisticProbe(n_features=8, lr=0.1, n_epochs=100, reg=0.01)
        probe.train(X.astype(np.float64), y.astype(np.float64))
        preds = probe.predict_proba(X.astype(np.float64))
        assert np.all(np.isfinite(preds)), "NaN/inf in predict_proba output"
        assert np.all((preds >= 0) & (preds <= 1)), "Probabilities outside [0, 1]"

    def test_predict_proba_range(self) -> None:
        """All predictions must lie in [0, 1] by sigmoid definition."""
        probe = exp.LogisticProbe(n_features=1)
        probe.w = np.array([100.0])  # extreme weight
        probe.b = 50.0
        X = np.linspace(-10, 10, 21).reshape(-1, 1)
        preds = probe.predict_proba(X)
        assert np.all((preds >= 0) & (preds <= 1))

    def test_epoch_log_structure(self) -> None:
        """epoch_log entries must have epoch, loss, train_auroc keys."""
        X, y = _small_xy(n=20, n_vars=1)
        probe = exp.LogisticProbe(n_features=1, n_epochs=150, lr=0.1)
        log = probe.train(X, y.astype(np.float64))
        assert len(log) > 0
        for entry in log:
            assert "epoch" in entry
            assert "loss" in entry
            assert "train_auroc" in entry


# ---------------------------------------------------------------------------
# _spline_jac_row
# ---------------------------------------------------------------------------


class TestSplineJacRow:
    def test_shape(self) -> None:
        jac = exp._spline_jac_row(0.3, n_knots=8)
        assert jac.shape == (8,)

    def test_exactly_two_nonzeros(self) -> None:
        """Linear interpolation activates exactly 2 knot basis functions."""
        jac = exp._spline_jac_row(0.3, n_knots=8)
        n_nonzero = int(np.sum(jac != 0))
        assert n_nonzero == 2, f"Expected 2 non-zeros, got {n_nonzero}"

    def test_sums_to_one(self) -> None:
        """Interpolation weights always sum to 1 (partition of unity)."""
        for x in [-0.9, 0.0, 0.5, 0.99]:
            jac = exp._spline_jac_row(x, n_knots=8)
            assert abs(float(jac.sum()) - 1.0) < 1e-6, f"x={x}: sum={jac.sum():.6f}"

    def test_boundary_clamping(self) -> None:
        """x outside [-1, 1] must be clamped without raising."""
        jac_low = exp._spline_jac_row(-2.0, n_knots=8)
        jac_high = exp._spline_jac_row(2.0, n_knots=8)
        assert jac_low.shape == (8,)
        assert jac_high.shape == (8,)
        assert abs(float(jac_low.sum()) - 1.0) < 1e-6
        assert abs(float(jac_high.sum()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _eval_energy
# ---------------------------------------------------------------------------


class TestEvalEnergy:
    def test_scalar_output(self) -> None:
        ctrl = np.ones((4, 8)) * 0.5
        x = np.zeros(4)
        e = exp._eval_energy(ctrl, x, n_knots=8)
        assert isinstance(e, float)

    def test_monotone_ctrl_gives_consistent_energy(self) -> None:
        """Monotone increasing ctrl should give higher energy for x=1 than x=-1."""
        ctrl = np.tile(np.linspace(0, 1, 8), (4, 1))  # shape (4, 8), monotone
        x_low = np.full(4, -1.0)
        x_high = np.full(4, 1.0)
        assert exp._eval_energy(ctrl, x_high, 8) >= exp._eval_energy(ctrl, x_low, 8)

    def test_zero_ctrl_gives_zero_energy(self) -> None:
        ctrl = np.zeros((4, 8))
        x = np.array([0.5, -0.5, 0.0, 0.3])
        assert abs(exp._eval_energy(ctrl, x, 8)) < 1e-8


# ---------------------------------------------------------------------------
# _enforce_mono
# ---------------------------------------------------------------------------


class TestEnforceMono:
    def test_output_non_decreasing(self) -> None:
        rng = np.random.default_rng(7)
        ctrl = rng.standard_normal((4, 8))
        ctrl_m = exp._enforce_mono(ctrl)
        for row in ctrl_m:
            diffs = np.diff(row)
            assert np.all(diffs >= -1e-8), f"Not non-decreasing: {row}"

    def test_min_is_zero(self) -> None:
        rng = np.random.default_rng(7)
        ctrl = rng.standard_normal((4, 8))
        ctrl_m = exp._enforce_mono(ctrl)
        for row in ctrl_m:
            assert abs(float(row.min())) < 1e-8

    def test_max_at_most_one(self) -> None:
        ctrl = np.tile(np.array([0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), (3, 1))
        ctrl_m = exp._enforce_mono(ctrl)
        for row in ctrl_m:
            assert float(row.max()) <= 1.0 + 1e-8


# ---------------------------------------------------------------------------
# _promote_grid
# ---------------------------------------------------------------------------


class TestPromoteGrid:
    def test_output_shape(self) -> None:
        ctrl_c = np.ones((4, 4))
        ctrl_f = exp._promote_grid(ctrl_c, n_fine=8)
        assert ctrl_f.shape == (4, 8)

    def test_constant_ctrl_stays_constant(self) -> None:
        """Promoting a flat control vector must remain flat."""
        ctrl_c = np.ones((3, 4)) * 0.5
        ctrl_f = exp._promote_grid(ctrl_c, n_fine=8)
        assert np.allclose(ctrl_f, 0.5, atol=1e-6)

    def test_boundary_values_preserved(self) -> None:
        """First and last knot values must be exactly preserved after promotion."""
        ctrl_c = np.array([[0.0, 0.25, 0.75, 1.0]])
        ctrl_f = exp._promote_grid(ctrl_c, n_fine=8)
        assert abs(float(ctrl_f[0, 0]) - 0.0) < 1e-6
        assert abs(float(ctrl_f[0, -1]) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# _nk_step
# ---------------------------------------------------------------------------


class TestNkStep:
    def test_output_shape(self) -> None:
        ctrl = np.zeros((4, 4))
        X_batch = np.zeros((3, 4))
        y_batch = np.array([1.0, 0.0, 1.0])
        ctrl_new = exp._nk_step(ctrl, X_batch, y_batch, n_knots=4, lam=1.0)
        assert ctrl_new.shape == (4, 4)

    def test_no_nan_with_regularisation(self) -> None:
        """Large λ prevents ill-conditioning; result must be finite."""
        rng = np.random.default_rng(42)
        ctrl = rng.normal(0, 0.1, (4, 4))
        X_batch = rng.uniform(-1, 1, (5, 4))
        y_batch = np.array([1.0, 0.0, 1.0, 0.0, 1.0])
        ctrl_new = exp._nk_step(ctrl, X_batch, y_batch, n_knots=4, lam=10.0)
        assert np.all(np.isfinite(ctrl_new)), "NaN/inf in NK step output"

    def test_gradient_clipping_applied(self) -> None:
        """With extreme residuals, the update step norm must be <= 1.0."""
        # Craft a case where the unclipped step would be enormous
        ctrl = np.zeros((2, 4))
        X_batch = np.array([[0.99, -0.99], [0.99, -0.99]])
        y_batch = np.array([1.0, 0.0])
        ctrl_new = exp._nk_step(ctrl, X_batch, y_batch, n_knots=4, lam=0.001)
        delta = ctrl_new - ctrl
        delta_norm = float(np.linalg.norm(delta.ravel()))
        assert delta_norm <= 1.0 + 1e-6, f"Gradient clip failed: ||Δw||={delta_norm:.4f}"


# ---------------------------------------------------------------------------
# Artifact schema validation
# ---------------------------------------------------------------------------


class TestArtifactSchema:
    REQUIRED_FIELDS = [
        ("n_pairs_used", int),
        ("auroc_thinkprm", float),
        ("auroc_gskan", float),
        ("auroc_nk_kaem", float),
        ("best_probe_auroc", float),
        ("best_probe_name", str),
        ("nk_convergence_speedup", float),
        ("gskan_auroc_vs_baseline", float),
        ("honest_verdict", str),
    ]

    VALID_VERDICTS = {
        "probes_trained_above_threshold",
        "partial_some_below_0.72",
        "blocked_insufficient_corpus",
        "failed",
    }

    @pytest.fixture
    def artifact(self) -> dict:
        path = (
            Path(__file__).resolve().parents[2]
            / "results"
            / "experiment_1045_probe_ensemble_v5.json"
        )
        if not path.exists():
            pytest.skip("Artifact not yet generated")
        return json.loads(path.read_text())

    def test_required_fields_present(self, artifact: dict) -> None:
        for field, _ in self.REQUIRED_FIELDS:
            assert field in artifact, f"Required field missing: {field}"

    def test_required_fields_correct_type(self, artifact: dict) -> None:
        for field, expected_type in self.REQUIRED_FIELDS:
            if field not in artifact:
                continue
            val = artifact[field]
            assert isinstance(val, expected_type), (
                f"Field '{field}': expected {expected_type.__name__}, got {type(val).__name__}"
            )

    def test_honest_verdict_valid(self, artifact: dict) -> None:
        verdict = artifact.get("honest_verdict", "")
        assert verdict in self.VALID_VERDICTS, (
            f"honest_verdict '{verdict}' not in {self.VALID_VERDICTS}"
        )

    def test_auroc_values_in_range(self, artifact: dict) -> None:
        for field in ["auroc_thinkprm", "auroc_gskan", "auroc_nk_kaem", "best_probe_auroc"]:
            val = artifact.get(field, -1)
            assert 0.0 <= val <= 1.0, f"{field}={val} outside [0, 1]"

    def test_n_pairs_used_positive(self, artifact: dict) -> None:
        assert artifact.get("n_pairs_used", 0) > 0

    def test_best_probe_name_matches_auroc(self, artifact: dict) -> None:
        """best_probe_auroc must equal the AUROC of best_probe_name."""
        name = artifact.get("best_probe_name", "")
        auroc_map = {
            "thinkprm": artifact.get("auroc_thinkprm", 0),
            "gskan": artifact.get("auroc_gskan", 0),
            "nk_kaem": artifact.get("auroc_nk_kaem", 0),
        }
        if name not in auroc_map:
            return  # non-standard name, skip
        expected = auroc_map[name]
        actual = artifact.get("best_probe_auroc", -1)
        assert abs(actual - expected) < 1e-4, (
            f"best_probe_auroc={actual} does not match {name}={expected}"
        )
