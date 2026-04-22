"""Tests for IAS gate calibration module.

Covers 100% of python/carnot/pipeline/ias_gate_calibration.py.

Spec: REQ-VERIFY-151, REQ-VERIFY-152, SCENARIO-VERIFY-200, SCENARIO-VERIFY-201
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from carnot.pipeline.ias_gate_calibration import (
    IASGateCalibration,
    QuantileRegressionHead,
    adaptive_gate_open,
    calibrate,
)


# ---------------------------------------------------------------------------
# REQ-VERIFY-151-1: QuantileRegressionHead.train()
# ---------------------------------------------------------------------------


class TestQuantileRegressionHead:
    """Tests for QuantileRegressionHead pinball loss minimisation."""

    def test_train_returns_10th_percentile(self):
        """REQ-VERIFY-151-1: train at q=0.10 returns the 10th percentile."""
        # 10 evenly-spaced values: 0.0, 0.1, ..., 0.9
        obs = [i / 10.0 for i in range(10)]
        head = QuantileRegressionHead()
        threshold = head.train(obs, quantile=0.10)
        # 10th percentile of [0.0, 0.1, ..., 0.9]: idx = 9 * 0.10 = 0.9 → 0.0*(0.1) + 0.1*(0.9) = 0.09
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 0.2

    def test_train_returns_median(self):
        """train at q=0.50 returns the median of the distribution."""
        obs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        head = QuantileRegressionHead()
        median = head.train(obs, quantile=0.50)
        # median of sorted [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        # idx = 5 * 0.50 = 2.5 → lo=2, hi=3, frac=0.5 → 0.4*0.5 + 0.6*0.5 = 0.5
        assert abs(median - 0.5) < 1e-9

    def test_train_single_observation(self):
        """SCENARIO: single observation returns that observation's value."""
        head = QuantileRegressionHead()
        # Any quantile of a single-value distribution is that value.
        assert head.train([0.42], quantile=0.10) == pytest.approx(0.42)
        assert head.train([0.42], quantile=0.90) == pytest.approx(0.42)

    def test_train_all_zeros(self):
        """SCENARIO: all-zero observations return threshold=0.0."""
        head = QuantileRegressionHead()
        assert head.train([0.0] * 20, quantile=0.10) == pytest.approx(0.0)

    def test_train_all_ones(self):
        """SCENARIO: all-one observations return threshold=1.0."""
        head = QuantileRegressionHead()
        assert head.train([1.0] * 20, quantile=0.10) == pytest.approx(1.0)

    def test_train_empty_raises(self):
        """train raises ValueError on empty observation list."""
        head = QuantileRegressionHead()
        with pytest.raises(ValueError, match="non-empty"):
            head.train([], quantile=0.10)

    def test_train_invalid_quantile_zero_raises(self):
        """train raises ValueError when quantile <= 0."""
        head = QuantileRegressionHead()
        with pytest.raises(ValueError, match="quantile"):
            head.train([0.5], quantile=0.0)

    def test_train_invalid_quantile_one_raises(self):
        """train raises ValueError when quantile >= 1."""
        head = QuantileRegressionHead()
        with pytest.raises(ValueError, match="quantile"):
            head.train([0.5], quantile=1.0)

    def test_train_high_variance_lower_threshold(self):
        """REQ-VERIFY-152-1: high-variance distribution → lower 10th-percentile threshold."""
        # High variance: values spread 0.0 to 1.0 → low 10th percentile.
        high_var = [i / 9.0 for i in range(10)]
        # Low variance: all near 0.5 → higher 10th percentile.
        low_var = [0.48 + (i * 0.004) for i in range(10)]
        head = QuantileRegressionHead()
        assert head.train(high_var, quantile=0.10) < head.train(low_var, quantile=0.10)

    def test_train_low_variance_higher_threshold(self):
        """REQ-VERIFY-152-2: low-variance distribution → higher 10th-percentile threshold."""
        tight = [0.9 + (i * 0.01) for i in range(10)]  # [0.90..0.99]
        wide = [i / 9.0 for i in range(10)]  # [0.0..1.0]
        head = QuantileRegressionHead()
        assert head.train(tight, quantile=0.10) > head.train(wide, quantile=0.10)


# ---------------------------------------------------------------------------
# IASGateCalibration dataclass
# ---------------------------------------------------------------------------


class TestIASGateCalibration:
    """Tests for IASGateCalibration dataclass construction."""

    def test_dataclass_fields(self):
        """REQ-VERIFY-151-3: IASGateCalibration has required fields."""
        cal = IASGateCalibration(
            symcode_threshold=0.05,
            structured_threshold=0.80,
            causal_threshold=0.03,
            calibrated_from_n=57,
        )
        assert cal.symcode_threshold == 0.05
        assert cal.structured_threshold == 0.80
        assert cal.causal_threshold == 0.03
        assert cal.calibrated_from_n == 57


# ---------------------------------------------------------------------------
# calibrate() — SCENARIO-VERIFY-200
# ---------------------------------------------------------------------------


def _make_fover_pairs(n: int = 10, correct_fraction: float = 0.5) -> list[dict]:
    """Generate synthetic FOVER pairs with a mix of correct/incorrect labels."""
    pairs = []
    for i in range(n):
        label = "correct" if i < int(n * correct_fraction) else "incorrect"
        pairs.append({
            "question_id": str(i),
            "step_text": f"step {i}",
            "label": label,
            "confidence": 0.9,
        })
    return pairs


class TestCalibrate:
    """Tests for calibrate() function."""

    def test_calibrate_returns_ias_gate_calibration(self):
        """SCENARIO-VERIFY-200: calibrate returns IASGateCalibration instance."""
        pairs = _make_fover_pairs(n=20, correct_fraction=0.5)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pairs, f)
            tmp_path = f.name
        try:
            cal = calibrate(tmp_path)
            assert isinstance(cal, IASGateCalibration)
        finally:
            os.unlink(tmp_path)

    def test_calibrate_n_equals_pair_count(self):
        """SCENARIO-VERIFY-200: calibrated_from_n == number of pairs in file."""
        pairs = _make_fover_pairs(n=57)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pairs, f)
            tmp_path = f.name
        try:
            cal = calibrate(tmp_path)
            assert cal.calibrated_from_n == 57
        finally:
            os.unlink(tmp_path)

    def test_calibrate_thresholds_in_unit_interval(self):
        """All calibrated thresholds are in [0, 1]."""
        pairs = _make_fover_pairs(n=20, correct_fraction=0.4)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pairs, f)
            tmp_path = f.name
        try:
            cal = calibrate(tmp_path)
            assert 0.0 <= cal.symcode_threshold <= 1.0
            assert 0.0 <= cal.structured_threshold <= 1.0
            assert 0.0 <= cal.causal_threshold <= 1.0
        finally:
            os.unlink(tmp_path)

    def test_calibrate_structured_threshold_higher_than_others(self):
        """Structured extractor fires on every step (confidence always used).

        When incorrect labels have confidence>0, structured threshold > symcode/causal
        because symcode/causal use 0.0 for incorrect steps.
        """
        # 50% incorrect with confidence=0.9 → symcode/causal have 0.0 for those
        # but structured always uses confidence.
        pairs = _make_fover_pairs(n=20, correct_fraction=0.5)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pairs, f)
            tmp_path = f.name
        try:
            cal = calibrate(tmp_path)
            # Structured always uses confidence; symcode/causal use 0.0 for incorrect.
            # So structured 10th percentile >= symcode/causal 10th percentile.
            assert cal.structured_threshold >= cal.symcode_threshold
            assert cal.structured_threshold >= cal.causal_threshold
        finally:
            os.unlink(tmp_path)

    def test_calibrate_all_correct_thresholds_equal_confidence(self):
        """All-correct FOVER pairs → symcode/causal thresholds equal confidence's 10th pct."""
        # All correct, confidence=1.0 → every extractor's distribution is [1.0]*n
        pairs = [
            {"question_id": str(i), "step_text": f"s{i}", "label": "correct", "confidence": 1.0}
            for i in range(10)
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(pairs, f)
            tmp_path = f.name
        try:
            cal = calibrate(tmp_path)
            assert cal.symcode_threshold == pytest.approx(1.0)
            assert cal.structured_threshold == pytest.approx(1.0)
            assert cal.causal_threshold == pytest.approx(1.0)
        finally:
            os.unlink(tmp_path)

    def test_calibrate_real_fover_pairs(self, tmp_path):
        """SCENARIO-VERIFY-200: calibrate works on the actual 57-pair live FOVER file."""
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        fover_path = repo_root / "results" / "fover_labeled_steps_live.json"
        if not fover_path.exists():
            pytest.skip("fover_labeled_steps_live.json not found")
        cal = calibrate(str(fover_path))
        assert cal.calibrated_from_n == 57
        assert 0.0 <= cal.symcode_threshold <= 1.0
        assert 0.0 <= cal.structured_threshold <= 1.0
        assert 0.0 <= cal.causal_threshold <= 1.0

    def test_calibrate_file_not_found_raises(self):
        """calibrate raises FileNotFoundError on missing file."""
        with pytest.raises(FileNotFoundError):
            calibrate("/nonexistent/path/fover.json")


# ---------------------------------------------------------------------------
# adaptive_gate_open() — REQ-VERIFY-152-3, SCENARIO-VERIFY-201
# ---------------------------------------------------------------------------


class TestAdaptiveGateOpen:
    """Tests for adaptive_gate_open()."""

    def _make_cal(
        self,
        sym: float = 0.10,
        struct: float = 0.10,
        causal: float = 0.10,
    ) -> IASGateCalibration:
        return IASGateCalibration(
            symcode_threshold=sym,
            structured_threshold=struct,
            causal_threshold=causal,
            calibrated_from_n=10,
        )

    def test_gate_opens_when_symcode_meets_threshold(self):
        """REQ-VERIFY-152-3: gate opens when symcode >= symcode_threshold."""
        cal = self._make_cal(sym=0.10, struct=0.50, causal=0.50)
        assert adaptive_gate_open(cal, symcode=0.10, structured=0.05, causal=0.05) is True

    def test_gate_opens_when_structured_meets_threshold(self):
        """REQ-VERIFY-152-3: gate opens when structured >= structured_threshold."""
        cal = self._make_cal(sym=0.50, struct=0.10, causal=0.50)
        assert adaptive_gate_open(cal, symcode=0.05, structured=0.10, causal=0.05) is True

    def test_gate_opens_when_causal_meets_threshold(self):
        """REQ-VERIFY-152-3: gate opens when causal >= causal_threshold."""
        cal = self._make_cal(sym=0.50, struct=0.50, causal=0.10)
        assert adaptive_gate_open(cal, symcode=0.05, structured=0.05, causal=0.10) is True

    def test_gate_closes_when_all_below_threshold(self):
        """REQ-VERIFY-152-3: gate closes when all extractors are below their thresholds."""
        cal = self._make_cal(sym=0.30, struct=0.30, causal=0.30)
        assert adaptive_gate_open(cal, symcode=0.05, structured=0.05, causal=0.05) is False

    def test_scenario_verify_201_exp50_recall_values(self):
        """SCENARIO-VERIFY-201: IAS gate opens for Exp .50 recalls where v3 gate closed.

        Uses calibration from the live FOVER file when present; falls back to a
        synthetic calibration where causal_threshold < 0.36 to validate the logic.
        """
        import pathlib
        repo_root = pathlib.Path(__file__).resolve().parents[2]
        fover_path = repo_root / "results" / "fover_labeled_steps_live.json"
        if fover_path.exists():
            cal = calibrate(str(fover_path))
        else:
            # Fallback: synthesise calibration where causal threshold is well below 0.36
            cal = IASGateCalibration(
                symcode_threshold=0.05,
                structured_threshold=0.80,  # structured=0.20 won't pass
                causal_threshold=0.05,      # causal=0.36 >> 0.05 → passes
                calibrated_from_n=57,
            )

        # .50 recall values: symcode=0.12, structured=0.20, causal=0.36
        result = adaptive_gate_open(cal, symcode=0.12, structured=0.20, causal=0.36)
        # At least one extractor should exceed its calibrated threshold.
        assert result is True

    def test_gate_or_logic_any_one_sufficient(self):
        """Gate uses OR logic: any single extractor passing is sufficient."""
        cal = self._make_cal(sym=0.05, struct=0.90, causal=0.90)
        # Only symcode passes its threshold (0.06 >= 0.05).
        assert adaptive_gate_open(cal, symcode=0.06, structured=0.05, causal=0.05) is True

    def test_gate_exactly_at_threshold_is_open(self):
        """Boundary: recall exactly equal to threshold → gate opens (>=, not >)."""
        cal = self._make_cal(sym=0.30, struct=0.30, causal=0.30)
        assert adaptive_gate_open(cal, symcode=0.30, structured=0.00, causal=0.00) is True

    def test_gate_just_below_threshold_is_closed(self):
        """Boundary: recall just below threshold → gate stays closed."""
        cal = self._make_cal(sym=0.30, struct=0.30, causal=0.30)
        assert (
            adaptive_gate_open(cal, symcode=0.2999, structured=0.2999, causal=0.2999) is False
        )
