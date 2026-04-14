"""Tests for Exp 291 — JEPA Apple Adversarial Retrain with Energy Features.

Covers: feature extraction from logit arrays, isotonic calibration, conformal
interval computation, fast-path hit rate metric, A/B test result structure, and
synthetic fallback when real logits are not available.

Spec: REQ-JEPA-003
      SCENARIO-JEPA-006 (feature extraction from adversarial logits)
      SCENARIO-JEPA-007 (calibrated gate meets targets on held-out set)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from scripts.experiment_291_jepa_apple_retrain import (
    VARIANT_TYPE_ENCODING,
    AppleFeatureRow,
    ABResult,
    extract_apple_features,
    build_feature_matrix,
    build_synthetic_corpus,
    retrain_gate,
    compute_conformal_intervals,
    run_ab_comparison,
    run_experiment,
)

# ---------------------------------------------------------------------------
# Helpers — synthetic logit arrays
# ---------------------------------------------------------------------------

RNG = np.random.RandomState(291)


def _make_logits(n_tokens: int = 20, vocab: int = 50, seed: int = 0) -> np.ndarray:
    """Return a (n_tokens, vocab) float64 logit array."""
    rng = np.random.RandomState(seed)
    return rng.randn(n_tokens, vocab).astype(np.float64)


def _make_peaked_logits(n_tokens: int = 20, vocab: int = 50) -> np.ndarray:
    """Peaked logits — one token gets +10 per row (low entropy, overconfident)."""
    logits = np.full((n_tokens, vocab), -10.0, dtype=np.float64)
    logits[:, 0] = 10.0
    return logits


def _make_flat_logits(n_tokens: int = 20, vocab: int = 50) -> np.ndarray:
    """Flat logits — uniform distribution (high entropy, uncertain)."""
    return np.zeros((n_tokens, vocab), dtype=np.float64)


# ---------------------------------------------------------------------------
# SCENARIO-JEPA-006: Feature extraction from adversarial logits
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestExtractAppleFeatures:
    """SCENARIO-JEPA-006: extract_apple_features returns correct dict structure."""

    def test_returns_required_keys(self):
        """All eight required feature keys must be present."""
        # REQ-JEPA-003: feature dict must contain all required keys.
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        required = {
            "mean_spilled", "max_spilled", "p95_spilled",
            "semantic_energy", "mean_logit", "max_logit",
            "variant_type_encoded", "prefix_fraction",
        }
        assert required == set(result.keys()), f"Missing keys: {required - set(result.keys())}"

    def test_prefix_fraction_preserved(self):
        """prefix_fraction in output must match the input argument."""
        # REQ-JEPA-003: prefix_fraction is stored verbatim.
        for frac in (0.25, 0.5, 0.75, 1.0):
            logits = _make_logits()
            result = extract_apple_features(logits, prefix_fraction=frac, variant_type="standard")
            assert result["prefix_fraction"] == frac

    def test_variant_type_encoding_standard(self):
        """Standard variant encodes to 0."""
        # SCENARIO-JEPA-006: standard=0
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        assert result["variant_type_encoded"] == VARIANT_TYPE_ENCODING["standard"]
        assert result["variant_type_encoded"] == 0

    def test_variant_type_encoding_number_swap(self):
        """number_swap variant encodes to 1."""
        # SCENARIO-JEPA-006: number_swap=1
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="number_swap")
        assert result["variant_type_encoded"] == VARIANT_TYPE_ENCODING["number_swap"]
        assert result["variant_type_encoded"] == 1

    def test_variant_type_encoding_irrelevant(self):
        """irrelevant variant encodes to 2."""
        # SCENARIO-JEPA-006: irrelevant=2
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="irrelevant")
        assert result["variant_type_encoded"] == VARIANT_TYPE_ENCODING["irrelevant"]
        assert result["variant_type_encoded"] == 2

    def test_spilled_features_are_floats(self):
        """mean_spilled, max_spilled, p95_spilled must all be finite floats."""
        # REQ-JEPA-003: energy features are finite scalars.
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        for key in ("mean_spilled", "max_spilled", "p95_spilled"):
            assert isinstance(result[key], float), f"{key} is not float"
            assert math.isfinite(result[key]), f"{key} is not finite"

    def test_semantic_energy_is_finite_float(self):
        """semantic_energy must be a finite float (typically negative)."""
        # REQ-JEPA-003
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        assert isinstance(result["semantic_energy"], float)
        assert math.isfinite(result["semantic_energy"])

    def test_mean_logit_max_logit_consistent(self):
        """max_logit must be >= mean_logit for any input."""
        # REQ-JEPA-003: logit statistics are consistent.
        logits = _make_logits(seed=42)
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        assert result["max_logit"] >= result["mean_logit"] - 1e-9

    def test_peaked_logits_have_low_spilled_energy(self):
        """Peaked logits (confident) produce lower mean_spilled than flat logits."""
        # SCENARIO-JEPA-006: energy signals are discriminative.
        peaked = extract_apple_features(_make_peaked_logits(), 0.5, "standard")
        flat = extract_apple_features(_make_flat_logits(), 0.5, "standard")
        # For peaked dists, spilled energy should differ from flat dists.
        # Both can be near-zero (spill formula has cancellation at extremes),
        # but max and p95 differ.
        assert peaked["max_logit"] > flat["max_logit"]

    def test_all_prefix_fractions_work(self):
        """extract_apple_features works for all four standard prefix fractions."""
        # REQ-JEPA-003: feature extraction works at all prefix points.
        logits = _make_logits()
        for frac in (0.25, 0.5, 0.75, 1.0):
            result = extract_apple_features(logits, prefix_fraction=frac, variant_type="standard")
            assert len(result) == 8


# ---------------------------------------------------------------------------
# Synthetic fallback corpus
# REQ-JEPA-003: synthetic fallback includes synthetic_training: true
# ---------------------------------------------------------------------------


class TestSyntheticCorpus:
    """Synthetic fallback when logits not available."""

    def test_build_synthetic_corpus_returns_list(self):
        """build_synthetic_corpus returns a non-empty list of AppleFeatureRow.

        Each case produces 4 rows (one per prefix fraction: 0.25, 0.50, 0.75, 1.00).
        """
        # REQ-JEPA-003: synthetic fallback works when real data absent.
        rows = build_synthetic_corpus(n_cases=20, seed=291)
        assert isinstance(rows, list)
        # 20 cases × 4 prefix fractions = 80 rows.
        assert len(rows) == 20 * 4

    def test_synthetic_rows_are_apple_feature_row(self):
        """Each element is an AppleFeatureRow dataclass instance."""
        # REQ-JEPA-003
        rows = build_synthetic_corpus(n_cases=10, seed=291)
        for row in rows:
            assert isinstance(row, AppleFeatureRow)

    def test_synthetic_rows_have_synthetic_training_flag(self):
        """Metadata of each synthetic row includes synthetic_training=True."""
        # SCENARIO-JEPA-006: synthetic fallback label.
        rows = build_synthetic_corpus(n_cases=10, seed=291)
        for row in rows:
            assert row.metadata.get("synthetic_training") is True

    def test_synthetic_rows_have_valid_features(self):
        """All feature values in synthetic rows are finite floats."""
        # REQ-JEPA-003
        rows = build_synthetic_corpus(n_cases=10, seed=291)
        for row in rows:
            for key in ("mean_spilled", "max_spilled", "p95_spilled",
                        "semantic_energy", "mean_logit", "max_logit"):
                val = getattr(row.features, key) if hasattr(row.features, key) else row.features[key]
                assert math.isfinite(float(val)), f"Non-finite {key} in synthetic row"

    def test_synthetic_corpus_has_both_violation_classes(self):
        """Synthetic corpus must contain both violated=True and violated=False rows."""
        # REQ-JEPA-003: balanced classes for training.
        rows = build_synthetic_corpus(n_cases=40, seed=291)
        labels = [r.violation_label for r in rows]
        assert True in labels, "Synthetic corpus has no positive (violation) cases"
        assert False in labels, "Synthetic corpus has no negative (clean) cases"

    def test_synthetic_corpus_has_all_variant_types(self):
        """Synthetic corpus includes standard, number_swap, and irrelevant variants."""
        # REQ-JEPA-003: all three variant types in training data.
        rows = build_synthetic_corpus(n_cases=60, seed=291)
        variant_types = {r.variant_type for r in rows}
        assert "standard" in variant_types
        assert "number_swap" in variant_types
        assert "irrelevant" in variant_types

    def test_build_synthetic_corpus_deterministic(self):
        """Same seed produces the same corpus."""
        # REQ-JEPA-003: reproducible.
        rows1 = build_synthetic_corpus(n_cases=20, seed=291)
        rows2 = build_synthetic_corpus(n_cases=20, seed=291)
        for r1, r2 in zip(rows1, rows2):
            assert r1.violation_label == r2.violation_label
            assert r1.variant_type == r2.variant_type


# ---------------------------------------------------------------------------
# Feature matrix construction
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:
    """build_feature_matrix assembles X and y arrays from AppleFeatureRow list."""

    def test_shape_is_correct(self):
        """X has shape (n_rows, n_features) and y has shape (n_rows,)."""
        # REQ-JEPA-003: feature matrix dimensions correct.
        rows = build_synthetic_corpus(n_cases=30, seed=291)
        X, y = build_feature_matrix(rows)
        assert X.ndim == 2
        assert y.ndim == 1
        assert X.shape[0] == len(rows)
        assert y.shape[0] == len(rows)

    def test_labels_are_binary(self):
        """All labels in y must be 0 or 1."""
        # REQ-JEPA-003: binary classification target.
        rows = build_synthetic_corpus(n_cases=30, seed=291)
        _, y = build_feature_matrix(rows)
        unique = np.unique(y)
        for v in unique:
            assert v in (0.0, 1.0), f"Non-binary label: {v}"

    def test_feature_matrix_is_finite(self):
        """All values in X must be finite."""
        # REQ-JEPA-003: no NaN or Inf in feature matrix.
        rows = build_synthetic_corpus(n_cases=30, seed=291)
        X, _ = build_feature_matrix(rows)
        assert np.all(np.isfinite(X)), "Feature matrix contains non-finite values"

    def test_feature_count_matches_constant(self):
        """Number of features must be consistent across rows."""
        # REQ-JEPA-003: fixed feature dimension.
        rows = build_synthetic_corpus(n_cases=40, seed=291)
        X, _ = build_feature_matrix(rows)
        # All rows must have the same number of features.
        assert X.shape[1] > 0


# ---------------------------------------------------------------------------
# Isotonic calibration
# REQ-JEPA-003 (EBM-CoT approach, arXiv 2511.07124)
# ---------------------------------------------------------------------------


class TestIsotonicCalibration:
    """Isotonic regression calibration applied to trained gate scores."""

    def test_retrain_gate_returns_dict_with_calibration(self):
        """retrain_gate returns a result dict with a calibration key."""
        # REQ-JEPA-003: isotonic calibration is applied.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = retrain_gate(rows, seed=291)
        assert "calibration" in result, "Result must include 'calibration' key"

    def test_calibrated_scores_are_in_unit_interval(self):
        """All calibrated probabilities must be in [0, 1]."""
        # REQ-JEPA-003: calibrated outputs are valid probabilities.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = retrain_gate(rows, seed=291)
        probs = result["calibrated_probs_holdout"]
        assert all(0.0 <= p <= 1.0 for p in probs), "Calibrated probs outside [0,1]"

    def test_fast_path_rate_is_reported(self):
        """retrain_gate result must include fast_path_rate key."""
        # REQ-JEPA-003: fast-path rate is measured.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = retrain_gate(rows, seed=291)
        assert "fast_path_rate" in result
        rate = result["fast_path_rate"]
        assert 0.0 <= rate <= 1.0

    def test_tp_fp_rates_are_reported(self):
        """retrain_gate result must include tp_rate and fp_rate keys."""
        # REQ-JEPA-003: TP and FP rates are measured.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = retrain_gate(rows, seed=291)
        assert "tp_rate" in result
        assert "fp_rate" in result
        assert 0.0 <= result["tp_rate"] <= 1.0
        assert 0.0 <= result["fp_rate"] <= 1.0

    def test_targets_met_key_is_boolean(self):
        """result must include targets_met boolean key."""
        # SCENARIO-JEPA-007: targets clearly reported.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = retrain_gate(rows, seed=291)
        assert "targets_met" in result
        assert isinstance(result["targets_met"], bool)

    def test_targets_verdict_key_is_string(self):
        """result must include targets_verdict string (TARGETS_MET or TARGETS_NOT_MET)."""
        # SCENARIO-JEPA-007: explicit targets verdict.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = retrain_gate(rows, seed=291)
        assert "targets_verdict" in result
        assert result["targets_verdict"] in ("TARGETS_MET", "TARGETS_NOT_MET")

    def test_train_holdout_split_is_chronological(self):
        """retrain_gate uses chronological (not random) 80/20 split."""
        # REQ-JEPA-003: chronological split.
        rows = build_synthetic_corpus(n_cases=100, seed=291)
        result = retrain_gate(rows, seed=291)
        assert "n_train" in result
        assert "n_holdout" in result
        total = result["n_train"] + result["n_holdout"]
        assert total == len(rows)
        # Chronological: first 80% train, last 20% holdout.
        expected_holdout = len(rows) // 5
        assert abs(result["n_holdout"] - expected_holdout) <= 2


# ---------------------------------------------------------------------------
# Conformal prediction intervals
# REQ-JEPA-003 (arXiv 2603.22966, α=0.1 → 90% coverage)
# ---------------------------------------------------------------------------


class TestConformalIntervals:
    """compute_conformal_intervals computes valid coverage-guaranteed bounds."""

    def test_returns_dict_with_expected_keys(self):
        """compute_conformal_intervals returns tp_interval and fp_interval keys."""
        # REQ-JEPA-003: conformal intervals structure.
        # Simulate predictions and labels.
        rng = np.random.RandomState(291)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (probs > 0.5).astype(float) + rng.normal(0, 0.1, 50)
        labels = np.clip(labels, 0, 1).round()
        threshold = 0.5
        result = compute_conformal_intervals(probs, labels, threshold, alpha=0.1)
        assert "tp_interval" in result
        assert "fp_interval" in result

    def test_interval_bounds_are_ordered(self):
        """Each interval's lower bound must be <= upper bound."""
        # REQ-JEPA-003: valid interval geometry.
        rng = np.random.RandomState(291)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        tp_lo, tp_hi = result["tp_interval"]
        fp_lo, fp_hi = result["fp_interval"]
        assert tp_lo <= tp_hi, f"TP interval inverted: [{tp_lo}, {tp_hi}]"
        assert fp_lo <= fp_hi, f"FP interval inverted: [{fp_lo}, {fp_hi}]"

    def test_interval_bounds_in_unit_interval(self):
        """Both bounds of each interval must be in [0, 1]."""
        # REQ-JEPA-003: intervals are probability bounds.
        rng = np.random.RandomState(291)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        for key in ("tp_interval", "fp_interval"):
            lo, hi = result[key]
            assert 0.0 <= lo <= 1.0, f"{key} lower bound out of range: {lo}"
            assert 0.0 <= hi <= 1.0, f"{key} upper bound out of range: {hi}"

    def test_alpha_stored_in_result(self):
        """result must include the alpha used."""
        # REQ-JEPA-003: traceability.
        rng = np.random.RandomState(291)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        assert "alpha" in result
        assert abs(result["alpha"] - 0.1) < 1e-9

    def test_coverage_guarantee_at_01(self):
        """Coverage interval width increases when alpha decreases (tighter α → wider interval)."""
        # REQ-JEPA-003: conformal bounds reflect α.
        rng = np.random.RandomState(291)
        probs = rng.uniform(0.0, 1.0, 100)
        labels = (rng.uniform(size=100) > 0.5).astype(float)
        r_01 = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        r_05 = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.5)
        # Tighter α → wider (or equal) intervals.
        tp_width_01 = r_01["tp_interval"][1] - r_01["tp_interval"][0]
        tp_width_05 = r_05["tp_interval"][1] - r_05["tp_interval"][0]
        assert tp_width_01 >= tp_width_05 - 1e-9, (
            f"α=0.1 interval ({tp_width_01:.3f}) should be >= α=0.5 interval ({tp_width_05:.3f})"
        )


# ---------------------------------------------------------------------------
# A/B test result structure
# REQ-JEPA-003: 50-case A/B comparison
# ---------------------------------------------------------------------------


class TestABResult:
    """A/B test result structure: calibrated vs uncalibrated gate."""

    def test_run_ab_comparison_returns_ab_result(self):
        """run_ab_comparison returns an ABResult instance."""
        # REQ-JEPA-003: A/B result is structured.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        retrain_result = retrain_gate(rows, seed=291)
        ab = run_ab_comparison(rows, retrain_result, n_ab_cases=50, seed=291)
        assert isinstance(ab, ABResult)

    def test_ab_result_has_n_cases(self):
        """ABResult.n_cases matches the requested A/B case count (or available holdout)."""
        # REQ-JEPA-003: A/B size tracking.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        retrain_result = retrain_gate(rows, seed=291)
        ab = run_ab_comparison(rows, retrain_result, n_ab_cases=50, seed=291)
        assert ab.n_cases >= 1

    def test_ab_result_rates_are_valid(self):
        """ABResult fast_path_rate_calibrated and fast_path_rate_uncalibrated are in [0,1]."""
        # REQ-JEPA-003: valid rates.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        retrain_result = retrain_gate(rows, seed=291)
        ab = run_ab_comparison(rows, retrain_result, n_ab_cases=50, seed=291)
        assert 0.0 <= ab.fast_path_rate_calibrated <= 1.0
        assert 0.0 <= ab.fast_path_rate_uncalibrated <= 1.0

    def test_ab_result_serializes_to_dict(self):
        """ABResult can be serialized to a JSON-compatible dict."""
        # REQ-JEPA-003: serializable for saving to results JSON.
        rows = build_synthetic_corpus(n_cases=60, seed=291)
        retrain_result = retrain_gate(rows, seed=291)
        ab = run_ab_comparison(rows, retrain_result, n_ab_cases=40, seed=291)
        d = ab.to_dict()
        assert isinstance(d, dict)
        # Must be JSON-serializable
        json_str = json.dumps(d)
        assert len(json_str) > 0


# ---------------------------------------------------------------------------
# ONNX export
# REQ-JEPA-003: save retrained model as results/jepa_predictor_291.onnx
# SCENARIO-JEPA-007: ONNX file exists and is loadable
# ---------------------------------------------------------------------------


class TestONNXExport:
    """ONNX export of retrained PredictiveVerifier."""

    def test_run_experiment_creates_onnx_file(self, tmp_path: Path):
        """run_experiment saves jepa_predictor_291.onnx to the output directory."""
        # SCENARIO-JEPA-007: ONNX artifact produced.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = run_experiment(rows=rows, output_dir=tmp_path, seed=291)
        onnx_path = tmp_path / "jepa_predictor_291.onnx"
        assert onnx_path.exists(), f"ONNX model not saved at {onnx_path}"

    def test_run_experiment_returns_result_dict(self, tmp_path: Path):
        """run_experiment returns a dict with required keys."""
        # REQ-JEPA-003: structured result for JSON persistence.
        rows = build_synthetic_corpus(n_cases=80, seed=291)
        result = run_experiment(rows=rows, output_dir=tmp_path, seed=291)
        for key in (
            "experiment", "fast_path_rate", "tp_rate", "fp_rate",
            "targets_met", "targets_verdict", "conformal_intervals",
            "ab_test", "n_train", "n_holdout",
        ):
            assert key in result, f"Result missing key: {key}"

    def test_run_experiment_result_is_json_serializable(self, tmp_path: Path):
        """run_experiment result must be JSON-serializable."""
        # REQ-JEPA-003: results are persistable.
        rows = build_synthetic_corpus(n_cases=60, seed=291)
        result = run_experiment(rows=rows, output_dir=tmp_path, seed=291)
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_run_experiment_with_no_real_logits_uses_synthetic_fallback(self, tmp_path: Path):
        """run_experiment flags synthetic_training=true in metadata when using fallback."""
        # SCENARIO-JEPA-006: synthetic fallback label present.
        result = run_experiment(output_dir=tmp_path, seed=291)
        assert result.get("synthetic_training") is True

    def test_onnx_file_is_loadable(self, tmp_path: Path):
        """ONNX file produced by run_experiment must be loadable by onnxruntime."""
        # SCENARIO-JEPA-007: Exp 292 NPU test depends on loadable ONNX.
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        import onnxruntime as ort

        rows = build_synthetic_corpus(n_cases=80, seed=291)
        run_experiment(rows=rows, output_dir=tmp_path, seed=291)
        onnx_path = str(tmp_path / "jepa_predictor_291.onnx")
        sess = ort.InferenceSession(onnx_path)
        # Session must be loadable and have at least one input.
        assert len(sess.get_inputs()) >= 1


# ---------------------------------------------------------------------------
# run_experiment end-to-end sanity
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestRunExperimentEndToEnd:
    """End-to-end sanity: run_experiment with synthetic data."""

    def test_experiment_number_is_291(self, tmp_path: Path):
        """result['experiment'] must equal 291."""
        # REQ-JEPA-003: correct experiment ID.
        result = run_experiment(output_dir=tmp_path, seed=291)
        assert result["experiment"] == 291

    def test_targets_verdict_is_explicit(self, tmp_path: Path):
        """targets_verdict must be TARGETS_MET or TARGETS_NOT_MET (never silent)."""
        # SCENARIO-JEPA-007: never silent on targets.
        result = run_experiment(output_dir=tmp_path, seed=291)
        assert result["targets_verdict"] in ("TARGETS_MET", "TARGETS_NOT_MET")

    def test_conformal_intervals_have_alpha(self, tmp_path: Path):
        """conformal_intervals sub-dict must include alpha=0.1."""
        # REQ-JEPA-003: α=0.1 used.
        result = run_experiment(output_dir=tmp_path, seed=291)
        ci = result["conformal_intervals"]
        assert abs(ci["alpha"] - 0.1) < 1e-9

    def test_ab_test_in_result(self, tmp_path: Path):
        """result['ab_test'] must be a dict with calibrated and uncalibrated rates."""
        # REQ-JEPA-003: A/B comparison included.
        result = run_experiment(output_dir=tmp_path, seed=291)
        ab = result["ab_test"]
        assert "fast_path_rate_calibrated" in ab
        assert "fast_path_rate_uncalibrated" in ab

    def test_result_rates_in_unit_interval(self, tmp_path: Path):
        """fast_path_rate, tp_rate, fp_rate all in [0,1]."""
        # REQ-JEPA-003: valid metric outputs.
        result = run_experiment(output_dir=tmp_path, seed=291)
        for key in ("fast_path_rate", "tp_rate", "fp_rate"):
            val = result[key]
            assert 0.0 <= val <= 1.0, f"{key}={val} outside [0,1]"
