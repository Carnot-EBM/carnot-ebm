"""Tests for PPSConstraintSelector and CouplingVarianceTracker (Exp 762).

Spec: REQ-LEARN-042, SCENARIO-LEARN-082
"""

from __future__ import annotations

import numpy as np
import pytest

from python.carnot.pipeline.pps_constraint_selector import (
    CouplingVarianceTracker,
    PPSConstraintSelector,
)


# ---------------------------------------------------------------------------
# CouplingVarianceTracker tests
# ---------------------------------------------------------------------------


class TestCouplingVarianceTracker:
    """Tests for CouplingVarianceTracker.  Spec: REQ-LEARN-042"""

    def test_get_variance_correct_rolling_variance(self):
        """get_variance returns correct rolling variance after updates.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=2, window_size=30)
        # Push known values: coupling 0 always 1.0, coupling 1 varies 0/1 alternating.
        for i in range(10):
            tracker.update(np.array([1.0, float(i % 2)], dtype=np.float64))

        variances = tracker.get_variance()
        # Coupling 0: all values are 1.0 → variance = 0.
        assert variances[0] == pytest.approx(0.0, abs=1e-9)
        # Coupling 1: alternating 0/1 → variance = 0.25.
        assert variances[1] == pytest.approx(0.25, abs=1e-9)

    def test_get_variance_single_observation_returns_zero(self):
        """Variance is 0.0 when fewer than 2 observations exist.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=3, window_size=30)
        tracker.update(np.array([5.0, -2.0, 0.0]))
        variances = tracker.get_variance()
        assert all(v == pytest.approx(0.0) for v in variances)

    def test_get_variance_respects_window_size(self):
        """Variance uses only the last window_size observations.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=1, window_size=3)
        # Push 5 values; only the last 3 should be in the window.
        for v in [100.0, 100.0, 0.0, 0.0, 0.0]:
            tracker.update(np.array([v]))
        # Window contains [0, 0, 0] → variance = 0.
        assert tracker.get_variance()[0] == pytest.approx(0.0, abs=1e-9)

    def test_get_frozen_mask_below_threshold(self):
        """get_frozen_mask returns True for couplings with variance < threshold.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=2, window_size=30)
        # Coupling 0: constant → variance 0 (< 0.01 → frozen).
        # Coupling 1: high variance → not frozen.
        for i in range(10):
            tracker.update(np.array([1.0, float(i)]))
        mask = tracker.get_frozen_mask(freeze_threshold=0.01)
        assert mask[0] is np.bool_(True) or bool(mask[0]) is True
        assert bool(mask[1]) is False

    def test_get_frozen_mask_above_threshold_not_frozen(self):
        """Couplings with variance >= threshold are NOT frozen.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=1, window_size=30)
        # Alternating 0/1 → variance 0.25 >> 0.01.
        for i in range(10):
            tracker.update(np.array([float(i % 2)]))
        mask = tracker.get_frozen_mask(freeze_threshold=0.01)
        assert bool(mask[0]) is False

    def test_get_frozen_mask_empty_window_returns_frozen(self):
        """An un-updated coupling (variance=0) is treated as frozen.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=2, window_size=30)
        mask = tracker.get_frozen_mask(freeze_threshold=0.01)
        # All variances are 0.0 (no observations) → all frozen.
        assert all(bool(m) is True for m in mask)


# ---------------------------------------------------------------------------
# PPSConstraintSelector tests
# ---------------------------------------------------------------------------


class TestPPSConstraintSelector:
    """Tests for PPSConstraintSelector.  Spec: REQ-LEARN-042"""

    def test_apply_mask_zeros_frozen_coupling_gradients(self):
        """apply_mask zeros gradient entries for frozen couplings.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=3, window_size=30)
        # Coupling 0: constant (frozen).  Coupling 1: high variance.  Coupling 2: constant (frozen).
        for i in range(10):
            tracker.update(np.array([1.0, float(i), 2.0]))

        selector = PPSConstraintSelector(tracker, freeze_threshold=0.01)
        gradient = np.array([5.0, 3.0, -2.0], dtype=np.float64)
        masked = selector.apply_mask(gradient)

        # Couplings 0 and 2 are frozen → gradient should be 0.
        assert masked[0] == pytest.approx(0.0)
        assert masked[2] == pytest.approx(0.0)
        # Coupling 1 is NOT frozen → gradient preserved.
        assert masked[1] == pytest.approx(3.0)

    def test_apply_mask_does_not_modify_input(self):
        """apply_mask returns a new array; the input gradient is unchanged.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=2, window_size=30)
        for _ in range(5):
            tracker.update(np.array([1.0, 1.0]))  # both frozen

        selector = PPSConstraintSelector(tracker, freeze_threshold=0.01)
        original = np.array([7.0, 8.0], dtype=np.float64)
        gradient = original.copy()
        selector.apply_mask(gradient)
        np.testing.assert_array_equal(gradient, original)

    def test_apply_mask_no_frozen_couplings_passes_gradient_unchanged(self):
        """When no couplings are frozen, apply_mask returns the gradient unchanged.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=2, window_size=30)
        # High variance for both couplings → neither frozen.
        for i in range(20):
            tracker.update(np.array([float(i % 2), float((i + 1) % 2)]))

        selector = PPSConstraintSelector(tracker, freeze_threshold=0.001)
        gradient = np.array([3.0, -1.5], dtype=np.float64)
        masked = selector.apply_mask(gradient)
        np.testing.assert_allclose(masked, gradient)

    def test_frozen_count_correct(self):
        """frozen_count returns the number of frozen couplings.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=4, window_size=30)
        # Couplings 0 and 2 constant (→ frozen), 1 and 3 varying.
        for i in range(10):
            tracker.update(np.array([1.0, float(i), 1.0, float(i * 2)]))

        selector = PPSConstraintSelector(tracker, freeze_threshold=0.01)
        assert selector.frozen_count() == 2

    def test_frozen_count_zero_when_all_high_variance(self):
        """frozen_count returns 0 when all couplings have high variance.

        Spec: REQ-LEARN-042
        """
        tracker = CouplingVarianceTracker(n_couplings=3, window_size=30)
        rng = np.random.default_rng(0)
        for _ in range(15):
            tracker.update(rng.standard_normal(3) * 5.0)  # large variance

        selector = PPSConstraintSelector(tracker, freeze_threshold=0.01)
        assert selector.frozen_count() == 0
