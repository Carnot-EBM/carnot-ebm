"""Tests for compute_retrain_verdict_v2 — 100% branch coverage.

Tests every branch of the four-outcome decision tree in
python/carnot/pipeline/fover_eorm_retrain.py::compute_retrain_verdict_v2.

Spec: REQ-LEARN-036, SCENARIO-LEARN-064, SCENARIO-LEARN-065
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _d in [str(_REPO_ROOT / "python")]:
    if _d not in sys.path:
        sys.path.insert(0, _d)

from carnot.pipeline.fover_eorm_retrain import compute_retrain_verdict_v2  # noqa: E402


# ---------------------------------------------------------------------------
# source='synthetic' → always 'synthetic_only' regardless of AUC or n_pairs
# ---------------------------------------------------------------------------


class TestSyntheticOnly:
    """SCENARIO-LEARN-065 (synthetic branch)."""

    def test_synthetic_source_returns_synthetic_only(self):
        # source != 'live' → synthetic_only regardless of other params
        assert compute_retrain_verdict_v2(0.4, 0.9, 100, "synthetic") == "synthetic_only"

    def test_synthetic_with_zero_pairs(self):
        assert compute_retrain_verdict_v2(0.5, 0.5, 0, "synthetic") == "synthetic_only"

    def test_non_live_source_arbitrary_string(self):
        # Any source other than 'live' returns 'synthetic_only'
        assert compute_retrain_verdict_v2(0.3, 0.8, 50, "simulated") == "synthetic_only"

    def test_empty_string_source(self):
        assert compute_retrain_verdict_v2(0.5, 0.6, 30, "") == "synthetic_only"


# ---------------------------------------------------------------------------
# source='live', n_real_pairs < 20 → 'real_data_insufficient'
# ---------------------------------------------------------------------------


class TestRealDataInsufficient:
    """SCENARIO-LEARN-065 (insufficient live pairs)."""

    def test_zero_pairs(self):
        assert compute_retrain_verdict_v2(0.5, 0.9, 0, "live") == "real_data_insufficient"

    def test_one_pair(self):
        assert compute_retrain_verdict_v2(0.4, 0.8, 1, "live") == "real_data_insufficient"

    def test_nineteen_pairs_boundary(self):
        # Boundary: 19 < 20 → insufficient
        assert compute_retrain_verdict_v2(0.4, 0.9, 19, "live") == "real_data_insufficient"

    def test_insufficient_even_when_auc_improves(self):
        # AUC improvement is irrelevant when n_pairs < 20
        assert compute_retrain_verdict_v2(0.4, 0.99, 5, "live") == "real_data_insufficient"

    def test_insufficient_even_when_auc_same(self):
        assert compute_retrain_verdict_v2(0.5, 0.5, 10, "live") == "real_data_insufficient"


# ---------------------------------------------------------------------------
# source='live', n_real_pairs >= 20, after_auc > before_auc → 'real_data_improvement'
# ---------------------------------------------------------------------------


class TestRealDataImprovement:
    """SCENARIO-LEARN-064: RETRO-024 closure path."""

    def test_twenty_pairs_boundary_with_improvement(self):
        # Boundary: exactly 20 pairs AND improvement → closes RETRO-024
        assert compute_retrain_verdict_v2(0.5, 0.51, 20, "live") == "real_data_improvement"

    def test_large_n_with_significant_improvement(self):
        assert compute_retrain_verdict_v2(0.5, 0.72, 57, "live") == "real_data_improvement"

    def test_minimal_improvement(self):
        # Infinitesimal improvement still counts
        assert compute_retrain_verdict_v2(0.5, 0.5 + 1e-9, 20, "live") == "real_data_improvement"

    def test_improvement_from_below_baseline(self):
        # before_auc below 0.5 is fine as long as after > before
        assert compute_retrain_verdict_v2(0.3, 0.31, 25, "live") == "real_data_improvement"


# ---------------------------------------------------------------------------
# source='live', n_real_pairs >= 20, after_auc <= before_auc → 'real_data_no_improvement'
# ---------------------------------------------------------------------------


class TestRealDataNoImprovement:
    """Honest negative: data was real, but AUC did not improve."""

    def test_equal_auc_is_no_improvement(self):
        # after_auc == before_auc → not > → no_improvement
        assert compute_retrain_verdict_v2(0.5, 0.5, 20, "live") == "real_data_no_improvement"

    def test_auc_regressed(self):
        # AUC going down → no_improvement
        assert compute_retrain_verdict_v2(0.6, 0.55, 30, "live") == "real_data_no_improvement"

    def test_large_n_no_improvement(self):
        assert compute_retrain_verdict_v2(0.75, 0.74, 100, "live") == "real_data_no_improvement"

    def test_twenty_pairs_exact_equal_auc(self):
        # Boundary: n=20 AND equal AUC
        assert compute_retrain_verdict_v2(0.5, 0.5, 20, "live") == "real_data_no_improvement"


# ---------------------------------------------------------------------------
# Priority order: source check before n_pairs check
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    """Ensure source='synthetic' short-circuits before n_pairs check."""

    def test_synthetic_with_many_pairs_still_synthetic_only(self):
        # Even n=1000 real pairs don't matter if source != 'live'
        assert compute_retrain_verdict_v2(0.5, 0.99, 1000, "synthetic") == "synthetic_only"

    def test_live_source_checked_before_n_pairs(self):
        # source='live' proceeds to n_pairs check, not short-circuited
        result = compute_retrain_verdict_v2(0.5, 0.6, 5, "live")
        assert result == "real_data_insufficient"
