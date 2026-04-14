"""Tests for Exp 299 — JEPA Real Logits Retrain.

Covers:
- Real logit loading from data/research/logits_294_*.npy + logits_295_*.npy;
  graceful fallback to synthetic when files are absent.
- training_source field: "real_logits" or "synthetic_fallback".
- 8-feature vector includes semantic_energy from SemanticEnergyExtractor (Exp 297).
- Isotonic calibration applied to output probabilities.
- Conformal Clopper-Pearson bounds α=0.1 present in results.
- ONNX export creates results/jepa_predictor_299.onnx.
- onnxruntime can load the exported model.
- comparison_vs_exp291 dict in results.

Spec: REQ-JEPA-003
      SCENARIO-JEPA-006 (feature extraction from adversarial logits)
      SCENARIO-JEPA-007 (calibrated gate meets targets on held-out set)
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

from scripts.experiment_299_jepa_real_logits import (
    EXPERIMENT_ID,
    FEATURE_NAMES,
    VARIANT_TYPE_ENCODING,
    AppleFeatureRow,
    build_feature_matrix,
    build_synthetic_corpus,
    compute_conformal_intervals,
    extract_apple_features,
    retrain_gate,
    run_ab_comparison,
    run_experiment,
    _load_logits_from_exp294_295,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic logit arrays
# ---------------------------------------------------------------------------


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
# Experiment constant checks
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestConstants:
    """Exp 299 uses experiment ID 299 and the correct 8-feature set."""

    def test_experiment_id_is_299(self):
        """EXPERIMENT_ID must be 299 for traceability."""
        # REQ-JEPA-003: correct experiment identifier.
        assert EXPERIMENT_ID == 299

    def test_feature_names_has_eight_entries(self):
        """FEATURE_NAMES must contain exactly 8 features."""
        # REQ-JEPA-003: fixed 8-feature dimensionality.
        assert len(FEATURE_NAMES) == 8

    def test_feature_names_includes_semantic_energy(self):
        """FEATURE_NAMES must include semantic_energy (Exp 297 extractor)."""
        # SCENARIO-JEPA-006: semantic_energy is a required feature.
        assert "semantic_energy" in FEATURE_NAMES

    def test_feature_names_includes_spilled_features(self):
        """FEATURE_NAMES must include mean_spilled, max_spilled, p95_spilled."""
        # REQ-JEPA-003: spilled energy features from SpilledEnergyExtractor.
        for fname in ("mean_spilled", "max_spilled", "p95_spilled"):
            assert fname in FEATURE_NAMES, f"Missing feature: {fname}"

    def test_feature_names_includes_logit_stats(self):
        """FEATURE_NAMES must include mean_logit and max_logit."""
        # REQ-JEPA-003: raw logit statistics.
        assert "mean_logit" in FEATURE_NAMES
        assert "max_logit" in FEATURE_NAMES

    def test_feature_names_includes_variant_and_prefix(self):
        """FEATURE_NAMES must include variant_type_encoded and prefix_fraction."""
        # REQ-JEPA-003: variant and prefix features.
        assert "variant_type_encoded" in FEATURE_NAMES
        assert "prefix_fraction" in FEATURE_NAMES


# ---------------------------------------------------------------------------
# Real logit loading — graceful fallback to synthetic
# REQ-JEPA-003: handles missing files gracefully
# ---------------------------------------------------------------------------


class TestRealLogitLoading:
    """_load_logits_from_exp294_295 returns None when no files are found."""

    def test_returns_none_when_no_files(self, tmp_path: Path):
        """Missing 294/295 logit files → returns None (triggers synthetic fallback)."""
        # REQ-JEPA-003: graceful fallback when real logits absent.
        result = _load_logits_from_exp294_295(tmp_path)
        assert result is None, "Expected None when no logit files present"

    def test_loads_294_npy_files(self, tmp_path: Path):
        """Saves logits_294_standard_0.npy → loaded as AppleFeatureRow list."""
        # REQ-JEPA-003: real logit loading from Exp 294 files.
        rng = np.random.RandomState(42)
        logits = rng.randn(20, 50).astype(np.float64)
        np.save(str(tmp_path / "logits_294_standard_0.npy"), logits)

        rows = _load_logits_from_exp294_295(tmp_path)
        assert rows is not None, "Expected rows when logit_294 file present"
        assert len(rows) > 0

    def test_loads_295_npy_files(self, tmp_path: Path):
        """Saves logits_295_verify_0.npy → loaded as AppleFeatureRow list."""
        # REQ-JEPA-003: real logit loading from Exp 295 files.
        rng = np.random.RandomState(43)
        logits = rng.randn(20, 50).astype(np.float64)
        np.save(str(tmp_path / "logits_295_verify_0.npy"), logits)

        rows = _load_logits_from_exp294_295(tmp_path)
        assert rows is not None
        assert len(rows) > 0

    def test_real_rows_have_synthetic_training_false(self, tmp_path: Path):
        """Rows loaded from real files must have synthetic_training=False in metadata."""
        # SCENARIO-JEPA-006: provenance tracking.
        rng = np.random.RandomState(44)
        logits = rng.randn(20, 50).astype(np.float64)
        np.save(str(tmp_path / "logits_294_standard_0.npy"), logits)

        rows = _load_logits_from_exp294_295(tmp_path)
        assert rows is not None
        for row in rows:
            assert row.metadata.get("synthetic_training") is False

    def test_returns_none_for_corrupt_files(self, tmp_path: Path):
        """A corrupt .npy file in the directory must not crash; returns None if no valid rows."""
        # REQ-JEPA-003: robustness to corrupt inputs.
        corrupt = tmp_path / "logits_294_bad.npy"
        corrupt.write_bytes(b"\x00\x01\x02garbage")

        result = _load_logits_from_exp294_295(tmp_path)
        # Either None (all files corrupt) or a valid list (other files loaded).
        assert result is None or isinstance(result, list)

    def test_variant_inference_number_swap(self, tmp_path: Path):
        """Filename containing 'number_swap' → variant_type='number_swap'."""
        # SCENARIO-JEPA-006: variant type inferred from filename.
        rng = np.random.RandomState(45)
        logits = rng.randn(20, 50).astype(np.float64)
        np.save(str(tmp_path / "logits_294_number_swap_0.npy"), logits)

        rows = _load_logits_from_exp294_295(tmp_path)
        assert rows is not None
        assert any(r.variant_type == "number_swap" for r in rows)

    def test_variant_inference_irrelevant(self, tmp_path: Path):
        """Filename containing 'irrelevant' → variant_type='irrelevant'."""
        # SCENARIO-JEPA-006: variant type inferred from filename.
        rng = np.random.RandomState(46)
        logits = rng.randn(20, 50).astype(np.float64)
        np.save(str(tmp_path / "logits_295_irrelevant_0.npy"), logits)

        rows = _load_logits_from_exp294_295(tmp_path)
        assert rows is not None
        assert any(r.variant_type == "irrelevant" for r in rows)

    def test_each_file_produces_four_rows(self, tmp_path: Path):
        """Each logit file produces 4 rows (one per prefix fraction)."""
        # REQ-JEPA-003: 4 prefix fractions per case.
        rng = np.random.RandomState(47)
        logits = rng.randn(20, 50).astype(np.float64)
        np.save(str(tmp_path / "logits_294_standard_0.npy"), logits)

        rows = _load_logits_from_exp294_295(tmp_path)
        assert rows is not None
        assert len(rows) == 4, f"Expected 4 rows per file, got {len(rows)}"


# ---------------------------------------------------------------------------
# training_source field
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestTrainingSourceField:
    """training_source in results is 'real_logits' or 'synthetic_fallback'."""

    def test_training_source_is_synthetic_fallback_when_no_real_data(self, tmp_path: Path):
        """run_experiment sets training_source='synthetic_fallback' when logits absent."""
        # REQ-JEPA-003: explicit provenance labelling.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert result["training_source"] == "synthetic_fallback"

    def test_training_source_is_real_logits_when_files_present(self, tmp_path: Path):
        """run_experiment sets training_source='real_logits' when 294/295 files present."""
        # REQ-JEPA-003: real data takes precedence over synthetic.
        rng = np.random.RandomState(48)
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        for i in range(10):
            logits = rng.randn(20, 50).astype(np.float64)
            # Mix violations (295) and baseline (294)
            label = "verify" if i % 2 == 0 else "standard"
            prefix = "295" if i % 2 == 0 else "294"
            np.save(str(data_dir / f"logits_{prefix}_{label}_{i}.npy"), logits)

        result = run_experiment(output_dir=tmp_path, data_dir=data_dir, seed=299)
        assert result["training_source"] == "real_logits"

    def test_training_source_is_string(self, tmp_path: Path):
        """training_source must be a string (not a bool or None)."""
        # REQ-JEPA-003: type safety.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert isinstance(result["training_source"], str)

    def test_training_source_is_one_of_two_values(self, tmp_path: Path):
        """training_source must be exactly 'real_logits' or 'synthetic_fallback'."""
        # REQ-JEPA-003: closed vocabulary.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert result["training_source"] in ("real_logits", "synthetic_fallback")


# ---------------------------------------------------------------------------
# 8-feature vector — semantic_energy from Exp 297 extractor
# SCENARIO-JEPA-006
# ---------------------------------------------------------------------------


class TestFeatureVectorSemanticEnergy:
    """8-feature vector includes semantic_energy from SemanticEnergyExtractor."""

    def test_extract_apple_features_returns_semantic_energy(self):
        """extract_apple_features result contains 'semantic_energy' key."""
        # SCENARIO-JEPA-006: semantic_energy is a first-class feature.
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        assert "semantic_energy" in result

    def test_semantic_energy_is_finite_float(self):
        """semantic_energy must be a finite float (typically negative)."""
        # SCENARIO-JEPA-006: valid semantic energy value.
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        assert isinstance(result["semantic_energy"], float)
        assert math.isfinite(result["semantic_energy"])

    def test_peaked_logits_have_more_negative_semantic_energy_than_flat(self):
        """Peaked logits produce more negative semantic_energy than flat logits."""
        # SCENARIO-JEPA-006: semantic energy discriminates confidence level.
        peaked = extract_apple_features(_make_peaked_logits(), 0.5, "standard")
        flat = extract_apple_features(_make_flat_logits(), 0.5, "standard")
        # Peaked → more confident → lower (more negative) semantic energy.
        assert peaked["semantic_energy"] < flat["semantic_energy"]

    def test_feature_vector_has_exactly_8_features(self):
        """extract_apple_features returns exactly 8 features."""
        # REQ-JEPA-003: fixed 8-feature dimensionality.
        logits = _make_logits()
        result = extract_apple_features(logits, prefix_fraction=0.5, variant_type="standard")
        assert len(result) == 8

    def test_feature_matrix_has_8_columns(self):
        """build_feature_matrix produces X with 8 columns."""
        # REQ-JEPA-003: feature matrix dimensionality.
        rows = build_synthetic_corpus(n_cases=20, seed=299)
        X, _ = build_feature_matrix(rows)
        assert X.shape[1] == 8, f"Expected 8 columns, got {X.shape[1]}"


# ---------------------------------------------------------------------------
# Isotonic calibration
# REQ-JEPA-003 (EBM-CoT approach, arXiv 2511.07124)
# ---------------------------------------------------------------------------


class TestIsotonicCalibration:
    """Isotonic calibration applied to holdout probabilities."""

    def test_calibrated_probs_are_in_unit_interval(self):
        """All calibrated holdout probabilities must be in [0, 1]."""
        # REQ-JEPA-003: valid probability outputs.
        rows = build_synthetic_corpus(n_cases=80, seed=299)
        result = retrain_gate(rows, seed=299)
        probs = result["calibrated_probs_holdout"]
        assert all(0.0 <= p <= 1.0 for p in probs), "Calibrated probs outside [0,1]"

    def test_calibration_method_is_isotonic_regression(self):
        """calibration.method must be 'isotonic_regression'."""
        # REQ-JEPA-003: isotonic calibration documented in output.
        rows = build_synthetic_corpus(n_cases=80, seed=299)
        result = retrain_gate(rows, seed=299)
        assert result["calibration"]["method"] == "isotonic_regression"

    def test_calibration_has_operating_threshold(self):
        """calibration dict must include operating_threshold."""
        # REQ-JEPA-003: threshold is reported.
        rows = build_synthetic_corpus(n_cases=80, seed=299)
        result = retrain_gate(rows, seed=299)
        assert "operating_threshold" in result["calibration"]

    def test_targets_met_is_bool(self):
        """retrain_gate result must have targets_met as bool."""
        # SCENARIO-JEPA-007: target verdict is explicit.
        rows = build_synthetic_corpus(n_cases=80, seed=299)
        result = retrain_gate(rows, seed=299)
        assert isinstance(result["targets_met"], bool)


# ---------------------------------------------------------------------------
# Conformal Clopper-Pearson bounds α=0.1
# REQ-JEPA-003 (arXiv 2603.22966)
# ---------------------------------------------------------------------------


class TestConformalIntervals:
    """compute_conformal_intervals uses Clopper-Pearson with α=0.1."""

    def test_returns_tp_and_fp_intervals(self):
        """compute_conformal_intervals returns tp_interval and fp_interval."""
        # REQ-JEPA-003: conformal interval structure.
        rng = np.random.RandomState(299)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        assert "tp_interval" in result
        assert "fp_interval" in result

    def test_alpha_is_stored_in_result(self):
        """result must include alpha=0.1."""
        # REQ-JEPA-003: traceability of α parameter.
        rng = np.random.RandomState(299)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        assert abs(result["alpha"] - 0.1) < 1e-9

    def test_interval_bounds_are_ordered(self):
        """Each interval must have lo <= hi."""
        # REQ-JEPA-003: valid interval geometry.
        rng = np.random.RandomState(299)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        tp_lo, tp_hi = result["tp_interval"]
        fp_lo, fp_hi = result["fp_interval"]
        assert tp_lo <= tp_hi
        assert fp_lo <= fp_hi

    def test_interval_bounds_in_unit_interval(self):
        """Both bounds of each interval must be in [0, 1]."""
        # REQ-JEPA-003: intervals are probability bounds.
        rng = np.random.RandomState(299)
        probs = rng.uniform(0.0, 1.0, 50)
        labels = (rng.uniform(size=50) > 0.5).astype(float)
        result = compute_conformal_intervals(probs, labels, threshold=0.5, alpha=0.1)
        for key in ("tp_interval", "fp_interval"):
            lo, hi = result[key]
            assert 0.0 <= lo <= 1.0
            assert 0.0 <= hi <= 1.0

    def test_run_experiment_result_has_conformal_intervals(self, tmp_path: Path):
        """run_experiment result must include conformal_intervals with alpha=0.1."""
        # REQ-JEPA-003: conformal bounds in final output.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        ci = result["conformal_intervals"]
        assert "tp_interval" in ci
        assert "fp_interval" in ci
        assert abs(ci["alpha"] - 0.1) < 1e-9


# ---------------------------------------------------------------------------
# ONNX export
# REQ-JEPA-003: save retrained model as results/jepa_predictor_299.onnx
# SCENARIO-JEPA-007
# ---------------------------------------------------------------------------


class TestONNXExport:
    """ONNX export creates jepa_predictor_299.onnx."""

    def test_run_experiment_creates_onnx_file(self, tmp_path: Path):
        """run_experiment saves jepa_predictor_299.onnx to output_dir."""
        # SCENARIO-JEPA-007: ONNX artifact produced.
        run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        onnx_path = tmp_path / "jepa_predictor_299.onnx"
        assert onnx_path.exists(), f"ONNX not at {onnx_path}"

    def test_onnx_path_in_result(self, tmp_path: Path):
        """result['onnx_path'] must point to jepa_predictor_299.onnx."""
        # REQ-JEPA-003: ONNX path is recorded in results.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert "onnx_path" in result
        assert "jepa_predictor_299" in result["onnx_path"]

    def test_onnx_file_is_loadable_by_onnxruntime(self, tmp_path: Path):
        """ONNX file produced must be loadable by onnxruntime."""
        # SCENARIO-JEPA-007: model is NPU/ORT-compatible.
        pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
        import onnxruntime as ort

        run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        onnx_path = str(tmp_path / "jepa_predictor_299.onnx")
        sess = ort.InferenceSession(onnx_path)
        assert len(sess.get_inputs()) >= 1

    def test_run_experiment_returns_serializable_result(self, tmp_path: Path):
        """run_experiment result must be JSON-serializable."""
        # REQ-JEPA-003: results are persistable.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        json_str = json.dumps(result)
        assert len(json_str) > 0

    def test_run_experiment_writes_results_json(self, tmp_path: Path):
        """run_experiment writes experiment_299_results.json to output_dir."""
        # REQ-JEPA-003: results file is created.
        run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        results_file = tmp_path / "experiment_299_results.json"
        assert results_file.exists()


# ---------------------------------------------------------------------------
# comparison_vs_exp291 dict
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestComparisonVsExp291:
    """comparison_vs_exp291 dict in results compares Exp 299 vs Exp 291 baseline."""

    def test_comparison_key_present(self, tmp_path: Path):
        """result must contain 'comparison_vs_exp291' key."""
        # REQ-JEPA-003: comparison with baseline experiment.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert "comparison_vs_exp291" in result

    def test_comparison_is_dict(self, tmp_path: Path):
        """comparison_vs_exp291 must be a dict."""
        # REQ-JEPA-003: structured comparison.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        cmp = result["comparison_vs_exp291"]
        assert isinstance(cmp, dict), f"Expected dict, got {type(cmp)}"

    def test_comparison_has_exp291_tp_rate(self, tmp_path: Path):
        """comparison_vs_exp291 must record the Exp 291 TP rate baseline."""
        # REQ-JEPA-003: baseline reference values.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        cmp = result["comparison_vs_exp291"]
        assert "exp291_tp_rate" in cmp

    def test_comparison_has_exp291_fp_rate(self, tmp_path: Path):
        """comparison_vs_exp291 must record the Exp 291 FP rate baseline."""
        # REQ-JEPA-003: baseline reference values.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        cmp = result["comparison_vs_exp291"]
        assert "exp291_fp_rate" in cmp

    def test_comparison_has_exp299_tp_and_fp_rates(self, tmp_path: Path):
        """comparison_vs_exp291 must record Exp 299 TP and FP rates."""
        # REQ-JEPA-003: current experiment values for comparison.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        cmp = result["comparison_vs_exp291"]
        assert "exp299_tp_rate" in cmp
        assert "exp299_fp_rate" in cmp

    def test_comparison_has_training_source_delta(self, tmp_path: Path):
        """comparison_vs_exp291 must note the training source."""
        # REQ-JEPA-003: provenance in comparison.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        cmp = result["comparison_vs_exp291"]
        assert "training_source" in cmp

    def test_comparison_values_are_floats_or_strings(self, tmp_path: Path):
        """All numeric comparison values must be floats in [0,1]."""
        # REQ-JEPA-003: valid metric values.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        cmp = result["comparison_vs_exp291"]
        for key in ("exp291_tp_rate", "exp291_fp_rate", "exp299_tp_rate", "exp299_fp_rate"):
            val = cmp[key]
            assert isinstance(val, float), f"{key} is not float: {type(val)}"
            assert 0.0 <= val <= 1.0, f"{key}={val} outside [0,1]"


# ---------------------------------------------------------------------------
# run_experiment end-to-end sanity
# REQ-JEPA-003
# ---------------------------------------------------------------------------


class TestRunExperimentEndToEnd:
    """End-to-end sanity for run_experiment."""

    def test_experiment_id_is_299(self, tmp_path: Path):
        """result['experiment'] must equal 299."""
        # REQ-JEPA-003: correct experiment identifier.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert result["experiment"] == 299

    def test_targets_verdict_is_explicit(self, tmp_path: Path):
        """targets_verdict must be TARGETS_MET or TARGETS_NOT_MET."""
        # SCENARIO-JEPA-007: never silent on targets.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert result["targets_verdict"] in ("TARGETS_MET", "TARGETS_NOT_MET")

    def test_fast_path_rate_in_unit_interval(self, tmp_path: Path):
        """fast_path_rate must be in [0, 1]."""
        # REQ-JEPA-003: valid metric.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert 0.0 <= result["fast_path_rate"] <= 1.0

    def test_tp_fp_rates_in_unit_interval(self, tmp_path: Path):
        """tp_rate and fp_rate must both be in [0, 1]."""
        # REQ-JEPA-003: valid metrics.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert 0.0 <= result["tp_rate"] <= 1.0
        assert 0.0 <= result["fp_rate"] <= 1.0

    def test_synthetic_fallback_reports_training_source(self, tmp_path: Path):
        """When data_dir is empty, training_source='synthetic_fallback'."""
        # SCENARIO-JEPA-006: synthetic fallback is labelled honestly.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert result["training_source"] == "synthetic_fallback"

    def test_result_has_required_keys(self, tmp_path: Path):
        """run_experiment result must contain all required top-level keys."""
        # REQ-JEPA-003: structured result for JSON persistence.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        for key in (
            "experiment", "fast_path_rate", "tp_rate", "fp_rate",
            "targets_met", "targets_verdict", "conformal_intervals",
            "n_train", "n_holdout", "training_source", "comparison_vs_exp291",
            "onnx_path", "feature_names",
        ):
            assert key in result, f"Result missing key: {key}"

    def test_feature_names_in_result_matches_constant(self, tmp_path: Path):
        """result['feature_names'] must equal FEATURE_NAMES."""
        # REQ-JEPA-003: feature list is persisted for reproducibility.
        result = run_experiment(output_dir=tmp_path, data_dir=tmp_path, seed=299)
        assert result["feature_names"] == FEATURE_NAMES
