"""Tests for Exp 515 helpers: compute_wilson_ci, is_statistically_positive.

100% coverage on python/carnot/pipeline/live_200q_v5_helpers.py.

Spec: REQ-BENCH-016, SCENARIO-BENCH-035, SCENARIO-BENCH-036
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from carnot.pipeline.live_200q_v5_helpers import (
    _erfinv_approx,
    compute_wilson_ci,
    is_statistically_positive,
)


# ---------------------------------------------------------------------------
# compute_wilson_ci — SCENARIO-BENCH-035
# ---------------------------------------------------------------------------


class TestComputeWilsonCI:
    """SCENARIO-BENCH-035: compute_wilson_ci returns correct bounds."""

    def test_zero_n_total(self):
        """n_total=0 → degenerate case returns (0.0, 0.0)."""
        lo, hi = compute_wilson_ci(0, 0)
        assert lo == 0.0 and hi == 0.0

    def test_scenario_bench_035_core(self):
        """SCENARIO-BENCH-035: n_successes=120, n_total=200 straddles 0.6."""
        lo, hi = compute_wilson_ci(120, 200)
        assert lo > 0.53, f"lower bound {lo} should exceed 0.53"
        assert hi < 0.67, f"upper bound {hi} should be below 0.67"
        assert lo < 0.6 < hi, f"CI [{lo}, {hi}] should straddle 0.6"
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0

    def test_all_correct_200q(self):
        lo, hi = compute_wilson_ci(200, 200)
        assert lo > 0.9 and hi == 1.0

    def test_none_correct_200q(self):
        lo, hi = compute_wilson_ci(0, 200)
        assert lo == 0.0 and hi < 0.02

    def test_bounds_in_unit_interval(self):
        for n_s, n_t in [(1, 1), (0, 1), (50, 100), (99, 100), (1, 200)]:
            lo, hi = compute_wilson_ci(n_s, n_t)
            assert 0.0 <= lo <= 1.0, f"lower={lo} out of range for ({n_s},{n_t})"
            assert 0.0 <= hi <= 1.0, f"upper={hi} out of range for ({n_s},{n_t})"
            assert lo <= hi, f"lower > upper for ({n_s},{n_t})"

    def test_lower_always_lte_upper(self):
        lo, hi = compute_wilson_ci(100, 200)
        assert lo <= hi

    def test_90_percent_ci_wider(self):
        """99% CI should be wider than 95% CI for the same data."""
        lo95, hi95 = compute_wilson_ci(100, 200, confidence=0.95)
        lo99, hi99 = compute_wilson_ci(100, 200, confidence=0.99)
        assert hi99 - lo99 > hi95 - lo95

    def test_90_percent_ci_narrower(self):
        """90% CI should be narrower than 95% CI for the same data."""
        lo90, hi90 = compute_wilson_ci(100, 200, confidence=0.90)
        lo95, hi95 = compute_wilson_ci(100, 200, confidence=0.95)
        assert hi90 - lo90 < hi95 - lo95

    def test_custom_confidence_fallback_path(self):
        """Confidence value not in z-table triggers the erfinv approximation."""
        lo, hi = compute_wilson_ci(100, 200, confidence=0.80)
        assert 0.0 <= lo <= 1.0 and 0.0 <= hi <= 1.0
        assert lo < 0.5 < hi

    def test_raises_negative_n_total(self):
        with pytest.raises(ValueError, match="n_total"):
            compute_wilson_ci(0, -1)

    def test_raises_negative_n_successes(self):
        with pytest.raises(ValueError, match="n_successes"):
            compute_wilson_ci(-1, 10)

    def test_raises_successes_gt_total(self):
        with pytest.raises(ValueError, match="n_successes"):
            compute_wilson_ci(11, 10)

    def test_raises_confidence_zero(self):
        with pytest.raises(ValueError, match="confidence"):
            compute_wilson_ci(5, 10, confidence=0.0)

    def test_raises_confidence_one(self):
        with pytest.raises(ValueError, match="confidence"):
            compute_wilson_ci(5, 10, confidence=1.0)

    def test_raises_confidence_gt_one(self):
        with pytest.raises(ValueError, match="confidence"):
            compute_wilson_ci(5, 10, confidence=1.5)

    def test_known_value_half_correct(self):
        """50/100 → CI straddles 0.5."""
        lo, hi = compute_wilson_ci(50, 100)
        assert lo < 0.5 < hi

    def test_n_total_one_success(self):
        lo, hi = compute_wilson_ci(1, 1)
        assert lo > 0.0 and hi == 1.0

    def test_n_total_one_failure(self):
        lo, hi = compute_wilson_ci(0, 1)
        assert lo == 0.0 and hi < 1.0


# ---------------------------------------------------------------------------
# _erfinv_approx — internal helper (covered transitively, also tested directly)
# ---------------------------------------------------------------------------


class TestErfinvApprox:
    def test_returns_float(self):
        result = _erfinv_approx(0.80)
        assert isinstance(result, float)

    def test_positive_for_positive_confidence(self):
        assert _erfinv_approx(0.80) > 0.0
        assert _erfinv_approx(0.95) > 0.0


# ---------------------------------------------------------------------------
# is_statistically_positive — SCENARIO-BENCH-036
# ---------------------------------------------------------------------------


class TestIsStatisticallyPositive:
    """SCENARIO-BENCH-036: is_statistically_positive iff lower bound > 0."""

    def test_positive_lower_returns_true(self):
        """SCENARIO-BENCH-036: lower bound > 0 → True."""
        assert is_statistically_positive(0.001) is True

    def test_large_positive_lower_returns_true(self):
        assert is_statistically_positive(0.5) is True

    def test_zero_lower_returns_false(self):
        """SCENARIO-BENCH-036: lower bound == 0 → False."""
        assert is_statistically_positive(0.0) is False

    def test_negative_lower_returns_false(self):
        """SCENARIO-BENCH-036: lower bound < 0 → False (defensive guard)."""
        assert is_statistically_positive(-0.001) is False

    def test_very_small_positive_lower_returns_true(self):
        assert is_statistically_positive(1e-10) is True

    def test_integrated_with_wilson_ci_positive_case(self):
        """End-to-end: 180/200 correct → lower bound well above 0."""
        lo, hi = compute_wilson_ci(180, 200)
        assert is_statistically_positive(lo) is True

    def test_integrated_with_wilson_ci_zero_correct(self):
        """End-to-end: 0/200 correct → lower bound is 0, not positive."""
        lo, hi = compute_wilson_ci(0, 200)
        assert is_statistically_positive(lo) is False
