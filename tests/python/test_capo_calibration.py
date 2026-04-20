"""Tests for CAPOCalibrationLoss.

Covers all branches of capo_calibration.py to achieve 100% coverage.

Spec: REQ-VERIFY-140, REQ-VERIFY-141,
      SCENARIO-VERIFY-171, SCENARIO-VERIFY-172, SCENARIO-VERIFY-173
"""

from __future__ import annotations

import pytest

from carnot.pipeline.capo_calibration import CAPOCalibrationLoss


class TestCAPOCalibrationLossInit:
    """REQ-VERIFY-140: CAPOCalibrationLoss initialises with expected defaults."""

    def test_default_lambda_and_margin(self):
        # SCENARIO-VERIFY-171: default lambda_cal=0.1, margin=1.0
        loss_fn = CAPOCalibrationLoss()
        assert loss_fn.lambda_cal == pytest.approx(0.1)
        assert loss_fn.margin == pytest.approx(1.0)

    def test_custom_lambda_and_margin(self):
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.3, margin=2.0)
        assert loss_fn.lambda_cal == pytest.approx(0.3)
        assert loss_fn.margin == pytest.approx(2.0)

    def test_calibration_threshold_constant(self):
        # The threshold that gates calibration term must be 0.3
        assert CAPOCalibrationLoss._CALIBRATION_THRESHOLD == pytest.approx(0.3)


class TestCAPOCalibrationLossComputeLoss:
    """REQ-VERIFY-141: compute_loss combines contrastive margin loss + WMW calibration."""

    def test_empty_lists_return_zero(self):
        # SCENARIO-VERIFY-173: empty input -> 0.0
        loss_fn = CAPOCalibrationLoss()
        assert loss_fn.compute_loss([], []) == pytest.approx(0.0)

    def test_empty_correct_returns_zero(self):
        loss_fn = CAPOCalibrationLoss()
        assert loss_fn.compute_loss([], [1.0, 2.0]) == pytest.approx(0.0)

    def test_empty_incorrect_returns_zero(self):
        loss_fn = CAPOCalibrationLoss()
        assert loss_fn.compute_loss([1.0, 2.0], []) == pytest.approx(0.0)

    def test_perfect_separation_no_calibration_penalty(self):
        # SCENARIO-VERIFY-172: when gap >= margin AND |diff| >= 0.3, both terms are zero.
        # scores_correct=0.0, scores_incorrect=2.0 => gap=2.0>=1.0, diff=-2.0 (|diff|=2.0>=0.3)
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.1, margin=1.0)
        result = loss_fn.compute_loss([0.0], [2.0])
        assert result == pytest.approx(0.0)

    def test_margin_loss_only_when_gap_small_but_diff_large(self):
        # gap = 0.0 (scores_incorrect - scores_correct = 0), diff = 0.0 => |diff| = 0 < 0.3
        # margin_loss = max(0, 1.0 - 0.0) = 1.0
        # cal_loss = (0.0 + 0.5)^2 = 0.25
        # total = 1.0 + 0.1 * 0.25 = 1.025
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.1, margin=1.0)
        result = loss_fn.compute_loss([1.0], [1.0])
        assert result == pytest.approx(1.0 + 0.1 * 0.25, rel=1e-6)

    def test_calibration_active_gap_zero_diff_zero(self):
        # With margin=0 (no contrastive term), just calibration:
        # diff = 1.0 - 1.0 = 0.0, |diff|=0 < 0.3 => cal_loss=(0+0.5)^2=0.25
        # total = 0 + 0.1 * 0.25 = 0.025
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.1, margin=0.0)
        result = loss_fn.compute_loss([1.0], [1.0])
        assert result == pytest.approx(0.1 * 0.25, rel=1e-6)

    def test_calibration_silent_when_diff_at_threshold(self):
        # diff = -0.3 exactly: |diff| = 0.3, NOT < 0.3, so calibration is silent.
        # scores_correct=0.7, scores_incorrect=1.0 => diff=0.7-1.0=-0.3
        # gap = 1.0-0.7 = 0.3 < margin(1.0), so margin_loss = 0.7
        # cal_loss = 0 (|diff| = 0.3 is NOT < 0.3)
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.1, margin=1.0)
        result = loss_fn.compute_loss([0.7], [1.0])
        # margin_loss = 1.0 - 0.3 = 0.7, cal term silent
        assert result == pytest.approx(0.7, rel=1e-6)

    def test_calibration_active_diff_within_threshold(self):
        # diff = 0.1 (correct score higher than incorrect — bad sign for EBM)
        # |diff| = 0.1 < 0.3, so calibration is ACTIVE
        # scores_correct=1.1, scores_incorrect=1.0 => diff=0.1
        # gap = 1.0-1.1 = -0.1, margin_loss = max(0, 1.0-(-0.1)) = 1.1
        # cal_loss = (0.1+0.5)^2 = 0.36
        # total = 1.1 + 0.1*0.36 = 1.136
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.1, margin=1.0)
        result = loss_fn.compute_loss([1.1], [1.0])
        assert result == pytest.approx(1.1 + 0.1 * 0.36, rel=1e-6)

    def test_batch_uses_min_length(self):
        # 3 correct, 2 incorrect: only 2 pairs used
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.0, margin=1.0)
        # Each pair: correct=0.0, incorrect=2.0 => gap=2.0, margin_loss=0, cal inactive
        result = loss_fn.compute_loss([0.0, 0.0, 0.0], [2.0, 2.0])
        assert result == pytest.approx(0.0)

    def test_mean_over_batch(self):
        # Pair 0: correct=0.0, incorrect=0.0 => margin_loss=1.0, diff=0, cal_loss=0.25
        #   pair0_total = 1.0 + 0.1*0.25 = 1.025
        # Pair 1: correct=0.0, incorrect=2.0 => margin_loss=0.0, diff=-2.0>=0.3, cal=0
        #   pair1_total = 0.0
        # mean = 1.025/2 = 0.5125
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.1, margin=1.0)
        result = loss_fn.compute_loss([0.0, 0.0], [0.0, 2.0])
        expected = (1.025 + 0.0) / 2
        assert result == pytest.approx(expected, rel=1e-6)

    def test_lambda_cal_zero_equals_pure_contrastive(self):
        # With lambda_cal=0, CAPO reduces to pure contrastive margin loss.
        # gap = 0.5, margin_loss = 0.5; cal term is 0 because lambda_cal=0.
        loss_fn = CAPOCalibrationLoss(lambda_cal=0.0, margin=1.0)
        result = loss_fn.compute_loss([1.0], [1.5])
        assert result == pytest.approx(0.5, rel=1e-6)

    def test_export_from_pipeline_init(self):
        # Verifies the class is exported from carnot.pipeline
        from carnot.pipeline import CAPOCalibrationLoss as Imported
        assert Imported is CAPOCalibrationLoss
