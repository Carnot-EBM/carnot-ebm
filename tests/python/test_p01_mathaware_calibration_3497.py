"""Unit tests for p01_mathaware_calibration (exp3497).

Traces to:
  REQ-KONA-3497 (MATH-aware energy-correctness calibration with distinct pipelines)
  SCENARIO-KONA-3497 (mathaware recalibration locates mechanism)
  SCENARIO-KONA-3497-BLOCKED (honest block on small subset)

All tests use small synthetic data so they run in milliseconds with no
live model or filesystem dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot.phase3.p01_mathaware_calibration import (  # noqa: E402
    MathAwareRecalibResult,
    distinct_pipeline_assert,
    math_aware_cv_auroc,
)


# ---------------------------------------------------------------------------
# distinct_pipeline_assert
# ---------------------------------------------------------------------------


def test_distinct_pipeline_assert_returns_true_when_different():
    # REQ-KONA-3497: arrays that differ in at least one position are distinct
    proc = [1.0, 2.0, 3.0]
    trained = [1.0, 2.0, 4.0]
    assert distinct_pipeline_assert(proc, trained) is True


def test_distinct_pipeline_assert_returns_false_when_identical():
    # SCENARIO-KONA-3497: identical arrays = pipeline-sharing bug
    proc = [1.0, 2.0, 3.0]
    trained = [1.0, 2.0, 3.0]
    assert distinct_pipeline_assert(proc, trained) is False


def test_distinct_pipeline_assert_trivially_true_for_different_lengths():
    # Different-length arrays are trivially distinct (no element-wise comparison).
    assert distinct_pipeline_assert([1.0], [1.0, 2.0]) is True


def test_distinct_pipeline_assert_trivially_true_for_empty():
    # Empty arrays — no comparison possible, return True (not a bug case).
    assert distinct_pipeline_assert([], []) is True


def test_distinct_pipeline_assert_uses_float_tolerance():
    # Values within floating-point tolerance are treated as equal.
    proc = [1.0, 2.0]
    trained = [1.0 + 1e-13, 2.0]  # within abs_tol=1e-12
    assert distinct_pipeline_assert(proc, trained) is False


def test_distinct_pipeline_assert_large_difference_is_true():
    # A meaningful difference should return True.
    proc = [0.0, 0.0]
    trained = [1.0, 1.0]
    assert distinct_pipeline_assert(proc, trained) is True


# ---------------------------------------------------------------------------
# math_aware_cv_auroc — degenerate cases
# ---------------------------------------------------------------------------


def _make_records(n_math: int, n_gsm: int) -> tuple[
    list[dict], list[list[list[float]]], list[list[int]]
]:
    """Make synthetic records, feats, labels for math_aware_cv_auroc tests."""
    records: list[dict] = []
    feats: list[list[list[float]]] = []
    labels: list[list[int]] = []

    for i in range(n_gsm):
        records.append({"problem_id": f"gsm8k-{i}", "gold": 1})
        # 3 candidates: 2 correct, 1 wrong — features differ
        feats.append([
            [0.1, 0.2, 0.3, 0.4, -0.5, 1.0],
            [0.2, 0.3, 0.4, 0.5, -0.6, 1.1],
            [0.9, 0.8, 0.7, 0.6, -2.0, 0.5],
        ])
        labels.append([1, 1, 0])

    for i in range(n_math):
        records.append({"problem_id": f"math-{i}", "gold": 42, "level": "L4"})
        feats.append([
            [0.1, 0.1, 0.1, 0.1, -0.5, 1.2],
            [0.8, 0.9, 0.8, 0.9, -1.5, 0.3],
            [0.7, 0.6, 0.5, 0.4, -2.0, 0.2],
        ])
        labels.append([1, 0, 0])

    return records, feats, labels


def test_math_aware_cv_auroc_returns_result_type():
    # REQ-KONA-3497: math_aware_cv_auroc returns MathAwareRecalibResult
    records, feats, labels = _make_records(n_math=10, n_gsm=10)
    result = math_aware_cv_auroc(records, feats, labels, seed=42)
    assert isinstance(result, MathAwareRecalibResult)


def test_math_aware_cv_auroc_auroc_in_range():
    # REQ-KONA-3497: AUROC must be in [0, 1]
    records, feats, labels = _make_records(n_math=10, n_gsm=10)
    result = math_aware_cv_auroc(records, feats, labels, seed=42)
    assert 0.0 <= result.mathaware_correctness_auroc <= 1.0


def test_math_aware_cv_auroc_counts_only_math_problems():
    # REQ-KONA-3497: only non-GSM8K records are used for recalibration
    records, feats, labels = _make_records(n_math=8, n_gsm=12)
    result = math_aware_cv_auroc(records, feats, labels, seed=42)
    assert result.n_math_problems == 8


def test_math_aware_cv_auroc_handles_no_math_problems():
    # SCENARIO-KONA-3497-BLOCKED: no MATH problems → returns degenerate result
    records, feats, labels = _make_records(n_math=0, n_gsm=10)
    result = math_aware_cv_auroc(records, feats, labels, seed=42)
    assert result.n_math_problems == 0
    assert result.mathaware_correctness_auroc == pytest.approx(0.5)
    assert result.n_folds_used == 0


def test_math_aware_cv_auroc_handles_one_math_problem():
    # With only 1 MATH problem, fewer folds than requested
    records, feats, labels = _make_records(n_math=1, n_gsm=5)
    result = math_aware_cv_auroc(records, feats, labels, seed=42, n_folds=5)
    # n_folds capped to min(5, 1) = 1; degenerate CV but should not crash
    assert isinstance(result.mathaware_correctness_auroc, float)
    assert result.n_math_problems == 1


def test_math_aware_cv_auroc_n_folds_caps_to_n_math():
    # If n_math < n_folds, actual folds = n_math
    records, feats, labels = _make_records(n_math=3, n_gsm=5)
    result = math_aware_cv_auroc(records, feats, labels, seed=42, n_folds=5)
    assert result.n_folds_used <= 3


def test_math_aware_cv_auroc_n_candidates_positive_when_math_exists():
    # REQ-KONA-3497: n_math_candidates > 0 when MATH problems exist
    records, feats, labels = _make_records(n_math=6, n_gsm=6)
    result = math_aware_cv_auroc(records, feats, labels, seed=42)
    assert result.n_math_candidates > 0
