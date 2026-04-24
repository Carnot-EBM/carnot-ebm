"""Tests for Exp 789: EBM Calibration Alignment.

Spec: REQ-CALIB-001, REQ-CALIB-002, SCENARIO-CALIB-001, SCENARIO-CALIB-002
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from python.carnot.pipeline.ebm_calibrator import (  # noqa: E402
    CalibrationBin,
    EBMCalibrator,
    _sigmoid,
)
from scripts.experiment_789_ebm_calibration_alignment import (  # noqa: E402
    classify_verdict,
    load_labeled_steps,
)


# ---------------------------------------------------------------------------
# _sigmoid helper
# ---------------------------------------------------------------------------


class TestSigmoid:
    """Internal helper used by EBMCalibrator confidence computation."""

    def test_sigmoid_zero(self):
        # sigmoid(0) == 0.5 by definition
        result = _sigmoid(np.array([0.0]))
        assert abs(result[0] - 0.5) < 1e-9

    def test_sigmoid_large_positive(self):
        # sigmoid(large) -> ~1.0
        result = _sigmoid(np.array([100.0]))
        assert result[0] > 0.99

    def test_sigmoid_large_negative(self):
        # sigmoid(-large) -> ~0.0
        result = _sigmoid(np.array([-100.0]))
        assert result[0] < 0.01

    def test_sigmoid_no_overflow(self):
        # Must not produce NaN for extreme values — REQ-CALIB-001 stability
        result = _sigmoid(np.array([-1000.0, 1000.0]))
        assert not np.any(np.isnan(result))


# ---------------------------------------------------------------------------
# EBMCalibrator.compute_ece
# ---------------------------------------------------------------------------


class TestComputeECE:
    """REQ-CALIB-001, SCENARIO-CALIB-001: ECE computation from energy bins."""

    def test_ece_returns_zero_for_perfect_calibration(self):
        # SCENARIO-CALIB-001: if confidence == accuracy in every bin, ECE=0.
        # We engineer this by making sigmoid(-e) == accuracy for each bin.
        # sigmoid(-e) == 0.5 when e == 0.  If all labels are 50% correct (half 0, half 1)
        # and all energies are 0, then bin_confidence = 0.5 and accuracy = 0.5.
        n = 100
        energies = [0.0] * n
        # Alternate correct/incorrect so each equal-freq bin has accuracy ~0.5
        labels = [i % 2 for i in range(n)]
        calibrator = EBMCalibrator(n_bins=10)
        ece = calibrator.compute_ece(energies, labels)
        # sigmoid(0) = 0.5, accuracy in each bin = 0.5 -> |0.5 - 0.5| = 0.0
        assert abs(ece) < 1e-9, f"Expected ECE=0.0 for perfect calibration, got {ece}"

    def test_ece_is_nonnegative(self):
        # ECE must always be >= 0 — it is a sum of absolute values. REQ-CALIB-001.
        rng = np.random.default_rng(7)
        energies = rng.standard_normal(60).tolist()
        labels = rng.integers(0, 2, 60).tolist()
        calibrator = EBMCalibrator(n_bins=10)
        ece = calibrator.compute_ece(energies, labels)
        assert ece >= 0.0

    def test_ece_is_at_most_one(self):
        # ECE is bounded by 1 since |accuracy - confidence| <= 1. REQ-CALIB-001.
        energies = [100.0] * 50 + [-100.0] * 50  # extreme energies
        labels = [1] * 50 + [0] * 50
        calibrator = EBMCalibrator(n_bins=10)
        ece = calibrator.compute_ece(energies, labels)
        assert ece <= 1.0

    def test_ece_raises_on_length_mismatch(self):
        # Mismatched energies/labels must raise ValueError. REQ-CALIB-001 data integrity.
        calibrator = EBMCalibrator(n_bins=10)
        with pytest.raises(ValueError, match="same length"):
            calibrator.compute_ece([1.0, 2.0], [1])

    def test_ece_empty_returns_zero(self):
        # Empty input: ECE defined as 0.0 to avoid division-by-zero. REQ-CALIB-001.
        calibrator = EBMCalibrator(n_bins=10)
        ece = calibrator.compute_ece([], [])
        assert ece == 0.0


# ---------------------------------------------------------------------------
# EBMCalibrator._build_bins (equal-frequency binning)
# ---------------------------------------------------------------------------


class TestBuildBins:
    """REQ-CALIB-001: equal-frequency binning produces n_bins equal-size bins."""

    def test_equal_frequency_binning_produces_n_bins(self):
        # SCENARIO-CALIB-001: must produce exactly n_bins bins.
        n = 100
        energies = list(range(n))
        labels = [i % 2 for i in range(n)]
        calibrator = EBMCalibrator(n_bins=10)
        bins = calibrator._build_bins(energies, labels)
        assert len(bins) == 10, f"Expected 10 bins, got {len(bins)}"

    def test_bins_are_equal_size(self):
        # Each bin should have the same number of samples when N is divisible by n_bins.
        n = 100
        energies = list(range(n))
        labels = [0] * n
        calibrator = EBMCalibrator(n_bins=10)
        bins = calibrator._build_bins(energies, labels)
        sizes = [b.n_samples for b in bins]
        assert all(s == 10 for s in sizes), f"Expected all bins size 10, got {sizes}"

    def test_last_bin_absorbs_remainder(self):
        # When N is not divisible by n_bins, last bin absorbs the remainder.
        n = 103
        energies = list(range(n))
        labels = [0] * n
        calibrator = EBMCalibrator(n_bins=10)
        bins = calibrator._build_bins(energies, labels)
        # First 9 bins have 10 samples, last bin has 13
        assert bins[-1].n_samples == 13
        assert sum(b.n_samples for b in bins) == n

    def test_bins_are_sorted_by_energy(self):
        # Bins must be in ascending energy order. REQ-CALIB-001.
        energies = [5.0, 1.0, 3.0, 2.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                    11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0]
        labels = [0] * 20
        calibrator = EBMCalibrator(n_bins=2)
        bins = calibrator._build_bins(energies, labels)
        assert bins[0].energy_low <= bins[1].energy_low

    def test_bin_confidence_in_valid_range(self):
        # bin_confidence = mean(sigmoid(-energy)) must be in [0, 1].
        rng = np.random.default_rng(42)
        energies = rng.standard_normal(100).tolist()
        labels = rng.integers(0, 2, 100).tolist()
        calibrator = EBMCalibrator(n_bins=10)
        bins = calibrator._build_bins(energies, labels)
        for b in bins:
            assert 0.0 <= b.bin_confidence <= 1.0

    def test_bin_accuracy_in_valid_range(self):
        # accuracy = mean(labels) must be in [0, 1].
        rng = np.random.default_rng(99)
        energies = rng.standard_normal(100).tolist()
        labels = rng.integers(0, 2, 100).tolist()
        calibrator = EBMCalibrator(n_bins=10)
        bins = calibrator._build_bins(energies, labels)
        for b in bins:
            assert 0.0 <= b.accuracy <= 1.0


# ---------------------------------------------------------------------------
# EBMCalibrator.fit_isotonic + ECE_after
# ---------------------------------------------------------------------------


class TestFitIsotonic:
    """REQ-CALIB-001, SCENARIO-CALIB-002: isotonic regression calibration."""

    def test_fit_isotonic_returns_sklearn_object(self):
        # fit_isotonic must return a fitted sklearn IsotonicRegression.
        from sklearn.isotonic import IsotonicRegression
        rng = np.random.default_rng(1)
        energies = rng.standard_normal(50).tolist()
        labels = rng.integers(0, 2, 50).tolist()
        calibrator = EBMCalibrator(n_bins=5)
        iso = calibrator.fit_isotonic(energies, labels)
        assert isinstance(iso, IsotonicRegression)

    def test_isotonic_predict_in_01(self):
        # Calibrated probabilities must be in [0, 1]. REQ-CALIB-001.
        rng = np.random.default_rng(2)
        energies = rng.standard_normal(60).tolist()
        labels = rng.integers(0, 2, 60).tolist()
        calibrator = EBMCalibrator(n_bins=6)
        iso = calibrator.fit_isotonic(energies, labels)
        probs = iso.predict(-np.array(energies))
        assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    def test_ece_improvement_formula(self):
        # REQ-CALIB-002: ece_improvement = ECE_before - ECE_after.
        # SCENARIO-CALIB-002: isotonic never worsens ECE on training data.
        rng = np.random.default_rng(3)
        energies = rng.standard_normal(100).tolist()
        labels = rng.integers(0, 2, 100).tolist()
        calibrator = EBMCalibrator(n_bins=10)
        ece_before = calibrator.compute_ece(energies, labels)
        iso = calibrator.fit_isotonic(energies, labels)
        probs = iso.predict(-np.array(energies)).tolist()
        ece_after = calibrator.compute_ece_from_probs(probs, labels)
        improvement = ece_before - ece_after
        # Isotonic on training data: ECE_after <= ECE_before (improvement >= 0)
        assert improvement >= -1e-9, (
            f"Isotonic must not worsen ECE on training data: "
            f"ECE_before={ece_before:.4f}, ECE_after={ece_after:.4f}"
        )

    def test_compute_ece_from_probs_raises_on_mismatch(self):
        # Length mismatch must raise ValueError. REQ-CALIB-001 data integrity.
        calibrator = EBMCalibrator(n_bins=5)
        with pytest.raises(ValueError, match="same length"):
            calibrator.compute_ece_from_probs([0.5, 0.6], [1])

    def test_compute_ece_from_probs_empty(self):
        calibrator = EBMCalibrator(n_bins=5)
        assert calibrator.compute_ece_from_probs([], []) == 0.0


# ---------------------------------------------------------------------------
# EBMCalibrator.save_curve
# ---------------------------------------------------------------------------


class TestSaveCurve:
    """REQ-CALIB-002: calibration curve saved to results/ebm_calibration_curve.json."""

    def test_save_curve_writes_json(self, tmp_path):
        # save_curve must write a valid JSON file. REQ-CALIB-002.
        bins = [
            CalibrationBin(
                energy_low=-1.0,
                energy_high=0.0,
                accuracy=0.8,
                n_samples=10,
                bin_confidence=0.75,
            )
        ]
        calibrator = EBMCalibrator()
        out = str(tmp_path / "curve.json")
        calibrator.save_curve(bins, out)
        with open(out) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["accuracy"] == 0.8
        assert data[0]["n_samples"] == 10

    def test_save_curve_all_bin_fields_present(self, tmp_path):
        # All CalibrationBin fields must appear in output JSON. REQ-CALIB-002.
        rng = np.random.default_rng(5)
        energies = rng.standard_normal(100).tolist()
        labels = rng.integers(0, 2, 100).tolist()
        calibrator = EBMCalibrator(n_bins=10)
        bins = calibrator.build_curve(energies, labels)
        out = str(tmp_path / "curve.json")
        calibrator.save_curve(bins, out)
        with open(out) as f:
            data = json.load(f)
        required_fields = {"energy_low", "energy_high", "accuracy", "n_samples", "bin_confidence"}
        for row in data:
            assert required_fields.issubset(row.keys())


# ---------------------------------------------------------------------------
# classify_verdict
# ---------------------------------------------------------------------------


class TestClassifyVerdict:
    """REQ-CALIB-002: honest_verdict classification logic."""

    def test_insufficient_data(self):
        assert classify_verdict(0.3, 0.2, 0.1, 15) == "insufficient_data"

    def test_energy_well_calibrated(self):
        assert classify_verdict(0.08, 0.05, 0.03, 50) == "energy_well_calibrated"

    def test_calibration_improved(self):
        assert classify_verdict(0.30, 0.20, 0.10, 50) == "calibration_improved"

    def test_calibration_marginal(self):
        assert classify_verdict(0.25, 0.23, 0.02, 50) == "calibration_marginal"

    def test_calibration_no_improvement(self):
        assert classify_verdict(0.25, 0.25, 0.0, 50) == "calibration_no_improvement"

    def test_calibration_no_improvement_negative(self):
        # Negative ece_improvement should also map to no_improvement.
        assert classify_verdict(0.25, 0.26, -0.01, 50) == "calibration_no_improvement"


# ---------------------------------------------------------------------------
# load_labeled_steps
# ---------------------------------------------------------------------------


class TestLoadLabeledSteps:
    """REQ-CALIB-001: load and normalize FoVer labeled steps."""

    def test_loads_v1_file(self, tmp_path):
        # Should load v1 correctly and convert "correct"/"incorrect" to 1/0.
        fover_dir = tmp_path / "results"
        fover_dir.mkdir()
        data = [
            {"step_text": "step A", "label": "correct"},
            {"step_text": "step B", "label": "incorrect"},
        ]
        (fover_dir / "fover_labeled_steps_live.json").write_text(json.dumps(data))
        # Temporarily patch _FOVER_V1 path via monkeypatching the repo root
        import scripts.experiment_789_ebm_calibration_alignment as mod
        orig_fover_v1 = mod._FOVER_V1
        orig_fover_v2 = mod._FOVER_V2
        mod._FOVER_V1 = "results/fover_labeled_steps_live.json"
        mod._FOVER_V2 = "results/fover_labeled_steps_live_v2.json"
        try:
            steps = load_labeled_steps(tmp_path)
        finally:
            mod._FOVER_V1 = orig_fover_v1
            mod._FOVER_V2 = orig_fover_v2
        assert len(steps) == 2
        assert steps[0] == ("step A", 1)
        assert steps[1] == ("step B", 0)

    def test_loads_v2_if_exists(self, tmp_path):
        # v2 steps should be pooled with v1. REQ-CALIB-001.
        fover_dir = tmp_path / "results"
        fover_dir.mkdir()
        v1 = [{"step_text": "A", "label": "correct"}]
        v2 = [{"step_text": "B", "label": "incorrect"}]
        (fover_dir / "fover_labeled_steps_live.json").write_text(json.dumps(v1))
        (fover_dir / "fover_labeled_steps_live_v2.json").write_text(json.dumps(v2))
        import scripts.experiment_789_ebm_calibration_alignment as mod
        orig_v1 = mod._FOVER_V1
        orig_v2 = mod._FOVER_V2
        mod._FOVER_V1 = "results/fover_labeled_steps_live.json"
        mod._FOVER_V2 = "results/fover_labeled_steps_live_v2.json"
        try:
            steps = load_labeled_steps(tmp_path)
        finally:
            mod._FOVER_V1 = orig_v1
            mod._FOVER_V2 = orig_v2
        assert len(steps) == 2

    def test_missing_v2_does_not_crash(self, tmp_path):
        # v2 is optional; missing file must be silently skipped. REQ-CALIB-001.
        fover_dir = tmp_path / "results"
        fover_dir.mkdir()
        v1 = [{"step_text": "A", "label": "correct"}]
        (fover_dir / "fover_labeled_steps_live.json").write_text(json.dumps(v1))
        import scripts.experiment_789_ebm_calibration_alignment as mod
        orig_v1 = mod._FOVER_V1
        orig_v2 = mod._FOVER_V2
        mod._FOVER_V1 = "results/fover_labeled_steps_live.json"
        mod._FOVER_V2 = "results/fover_labeled_steps_live_v2.json"
        try:
            steps = load_labeled_steps(tmp_path)
        finally:
            mod._FOVER_V1 = orig_v1
            mod._FOVER_V2 = orig_v2
        assert len(steps) == 1
