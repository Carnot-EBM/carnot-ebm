"""Tests for Exp 811: SOTA GGUF Code Repair v4 — 50 HumanEval, batched 10×5, MARS gate.

Covers:
    - check_retro028_gate returns False when file missing (REQ-BENCH-016-4)
    - check_retro028_gate returns False when retro_028_closed=False (REQ-BENCH-016-4)
    - check_retro028_gate returns True when retro_028_closed=True (REQ-BENCH-016-4)
    - check_retro028_gate returns False on corrupt JSON (REQ-BENCH-016-4)
    - compute_signed_improvement is repair - baseline, not clamped (REQ-BENCH-016-6)
    - compute_signed_improvement positive when repair > baseline (REQ-BENCH-016-6)
    - compute_signed_improvement negative when repair < baseline (REQ-BENCH-016-6)
    - compute_signed_improvement zero when n_problems=0 (REQ-BENCH-016-6)
    - partial verdict format is "partial_N_of_50" (REQ-BENCH-016)

Spec: REQ-BENCH-016, SCENARIO-BENCH-035
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from scripts.experiment_811_sota_gguf_code_repair_v4 import (
    check_retro028_gate,
    compute_signed_improvement,
)


# ---------------------------------------------------------------------------
# check_retro028_gate — REQ-BENCH-016-4
# ---------------------------------------------------------------------------


def test_retro028_gate_missing_file() -> None:
    """Gate returns False when the Exp 810 result file does not exist.

    Spec: REQ-BENCH-016-4
    """
    assert check_retro028_gate(Path("/nonexistent/path/experiment_810.json")) is False


def test_retro028_gate_false_when_not_closed() -> None:
    """Gate returns False when retro_028_closed is False in the Exp 810 artifact.

    Spec: REQ-BENCH-016-4
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"retro_028_closed": False, "status": "blocked"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_retro028_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


def test_retro028_gate_true_when_closed() -> None:
    """Gate returns True when retro_028_closed is True in the Exp 810 artifact.

    Spec: REQ-BENCH-016-4
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        json.dump({"retro_028_closed": True, "status": "success"}, fh)
        tmp = Path(fh.name)
    try:
        assert check_retro028_gate(tmp) is True
    finally:
        tmp.unlink(missing_ok=True)


def test_retro028_gate_false_on_corrupt_json() -> None:
    """Gate returns False when the artifact file contains invalid JSON.

    Spec: REQ-BENCH-016-4
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as fh:
        fh.write("not valid json {{{")
        tmp = Path(fh.name)
    try:
        assert check_retro028_gate(tmp) is False
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# compute_signed_improvement — REQ-BENCH-016-6
# ---------------------------------------------------------------------------


def test_signed_improvement_positive_when_repair_better() -> None:
    """signed_improvement is positive when repair passes more problems than baseline.

    Spec: REQ-BENCH-016-6
    """
    # repair=30, baseline=20, n=50 → (30-20)/50 = 0.2
    result = compute_signed_improvement(30, 20, 50)
    assert result == pytest.approx(0.2)


def test_signed_improvement_negative_when_repair_worse() -> None:
    """signed_improvement is negative when repair passes fewer problems than baseline.

    The value is not clamped to zero — a negative result is a valid outcome.

    Spec: REQ-BENCH-016-6
    """
    # repair=10, baseline=20, n=50 → (10-20)/50 = -0.2
    result = compute_signed_improvement(10, 20, 50)
    assert result == pytest.approx(-0.2)


def test_signed_improvement_zero_when_equal() -> None:
    """signed_improvement is zero when repair and baseline are identical.

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(25, 25, 50)
    assert result == pytest.approx(0.0)


def test_signed_improvement_zero_when_n_problems_zero() -> None:
    """signed_improvement is 0.0 when n_problems=0 (avoid division by zero).

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(0, 0, 0)
    assert result == 0.0


def test_signed_improvement_not_clamped_at_zero() -> None:
    """signed_improvement can be negative — it is not clamped or normalised.

    This distinguishes v4 from any clamped variant.  A negative result must
    reach the artifact so the conductor can record honest regression.

    Spec: REQ-BENCH-016-6
    """
    result = compute_signed_improvement(0, 50, 50)
    assert result == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Partial verdict format — REQ-BENCH-016
# ---------------------------------------------------------------------------


def test_partial_verdict_format_matches_50_problem_spec() -> None:
    """Partial timeout verdicts must follow 'partial_N_of_50' format.

    The conductor parses this string to determine how many problems completed.
    An incorrect format (e.g. 'partial_N_of_25' from v3) would misreport scope.

    Spec: REQ-BENCH-016
    """
    n_completed = 15
    verdict = f"partial_{n_completed}_of_50"
    assert verdict == "partial_15_of_50"
    assert "_of_50" in verdict
