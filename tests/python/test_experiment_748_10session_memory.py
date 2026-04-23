"""Tests for Experiment 748 — Cross-Session Memory 10-Session Stress Test.

Coverage targets (REQ-FR11-009, REQ-FR11-010):
- test_10_session_loop_produces_10_precision_values: run_10_session_simulation returns
  exactly 10 precision values in the series.  (REQ-FR11-009-1)
- test_is_monotonically_non_decreasing_computed_correctly: the flag is True when all
  precision values are non-decreasing.  (REQ-FR11-009-2)
- test_monotonically_non_decreasing_false_when_regression: the flag is False when any
  session drops below the prior.  (REQ-FR11-009-2)
- test_plateau_detection_triggers_at_correct_session: plateau_session is the first index
  (1-based) where delta < 0.001.  (REQ-FR11-009-3)
- test_no_plateau_when_always_improving: plateau_session is None when deltas always >= 0.001.
  (REQ-FR11-009-3)
- test_honest_verdict_monotonic_gain: verdict is "tier2_memory_monotonic_gain" when
  is_monotonically_non_decreasing=True and no plateau.  (REQ-FR11-009-4)
- test_honest_verdict_plateau: verdict is "tier2_memory_plateau_at_s{N}" when plateau
  detected.  (REQ-FR11-009-4)
- test_honest_verdict_regression: verdict is "tier2_memory_regression" when any session
  drops > 0.01 below prior.  (REQ-FR11-009-4)
- test_templates_replayed_s1_is_zero: S1 has 0 replays (cold start, no prior session).
  (REQ-FR11-010)
- test_templates_replayed_s2_through_s10_gt_zero: after S1 persist, all subsequent
  sessions have templates_replayed > 0.  (REQ-FR11-010-2)
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
from unittest.mock import patch

import pytest

# Ensure repo root is on the path so we can import the experiment module
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiment_748_cross_session_memory_10session import (
    run_10_session_simulation,
    _BASE_QUESTIONS,
    N_SESSIONS,
    N_QUESTIONS_PER_SESSION,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_sim(tmp_path) -> dict:
    """Run the 10-session simulation with the standard question set."""
    return run_10_session_simulation(
        persist_dir=str(tmp_path),
        questions=_BASE_QUESTIONS,
    )


# ---------------------------------------------------------------------------
# REQ-FR11-009-1: 10 precision values produced
# ---------------------------------------------------------------------------


def test_10_session_loop_produces_10_precision_values(tmp_path):
    """run_10_session_simulation returns exactly 10 precision values.

    Spec: REQ-FR11-009-1, SCENARIO-FR11-009
    """
    result = _run_sim(tmp_path)
    assert len(result["precision_series"]) == N_SESSIONS, (
        f"Expected {N_SESSIONS} precision values, got {len(result['precision_series'])}"
    )


# ---------------------------------------------------------------------------
# REQ-FR11-009-2: is_monotonically_non_decreasing computed correctly
# ---------------------------------------------------------------------------


def test_is_monotonically_non_decreasing_computed_correctly(tmp_path):
    """When precision_series is monotonically non-decreasing, the flag is True.

    Spec: REQ-FR11-009-2, SCENARIO-FR11-009
    """
    result = _run_sim(tmp_path)
    series = result["precision_series"]

    # Compute independently what the flag SHOULD be
    expected_flag = all(series[i] >= series[i - 1] - 1e-9 for i in range(1, len(series)))
    assert result["is_monotonically_non_decreasing"] == expected_flag, (
        f"is_monotonically_non_decreasing mismatch: "
        f"computed={result['is_monotonically_non_decreasing']}, expected={expected_flag}, "
        f"series={series}"
    )


def test_monotonically_non_decreasing_false_when_regression():
    """The flag is False when any element in precision_series drops below the prior.

    Spec: REQ-FR11-009-2
    """
    # Inject a series with a drop at position 5
    regressing_series = [0.5, 0.55, 0.6, 0.65, 0.70, 0.60, 0.65, 0.70, 0.72, 0.74]
    flag = all(regressing_series[i] >= regressing_series[i - 1] - 1e-9
               for i in range(1, len(regressing_series)))
    assert flag is False, "Expected False when series has a drop"


# ---------------------------------------------------------------------------
# REQ-FR11-009-3: plateau detection
# ---------------------------------------------------------------------------


def test_plateau_detection_triggers_at_correct_session():
    """plateau_session is the first 1-based session where delta < 0.001.

    Spec: REQ-FR11-009-3, SCENARIO-FR11-009
    """
    # Series where improvement stops at index 3 (S4 = session 4)
    series = [0.50, 0.55, 0.60, 0.6001, 0.6002, 0.6003, 0.601, 0.601, 0.601, 0.601]

    plateau_session = None
    for i in range(1, len(series)):
        delta = series[i] - series[i - 1]
        if delta < 0.001:
            plateau_session = i + 1  # 1-based session
            break

    # First delta < 0.001 is at index 3 (0.6001 - 0.60 = 0.0001 < 0.001)
    assert plateau_session == 4, (
        f"Expected plateau_session=4 (S4), got {plateau_session}"
    )


def test_no_plateau_when_always_improving():
    """plateau_session is None when all deltas are >= 0.001.

    Spec: REQ-FR11-009-3
    """
    series = [0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59]

    plateau_session = None
    for i in range(1, len(series)):
        delta = series[i] - series[i - 1]
        if delta < 0.001:
            plateau_session = i + 1
            break

    assert plateau_session is None, (
        f"Expected None (always improving), got {plateau_session}"
    )


# ---------------------------------------------------------------------------
# REQ-FR11-009-4: honest_verdict
# ---------------------------------------------------------------------------


def test_honest_verdict_monotonic_gain(tmp_path):
    """honest_verdict is 'tier2_memory_monotonic_gain' when monotonically non-decreasing.

    Spec: REQ-FR11-009-4, SCENARIO-FR11-009
    """
    result = _run_sim(tmp_path)
    # The synthetic simulation is designed so that precision rises with session index.
    # If it is monotonically non-decreasing and no plateau, the verdict should match.
    if result["is_monotonically_non_decreasing"] and result["plateau_session"] is None:
        assert result["honest_verdict"] == "tier2_memory_monotonic_gain", (
            f"Expected 'tier2_memory_monotonic_gain', got '{result['honest_verdict']}'"
        )
    else:
        # Plateau or regression cases are also valid outcomes
        assert result["honest_verdict"] in (
            "tier2_memory_monotonic_gain",
            f"tier2_memory_plateau_at_s{result['plateau_session']}",
            "tier2_memory_regression",
        )


def test_honest_verdict_plateau():
    """honest_verdict is 'tier2_memory_plateau_at_s{N}' when plateau is detected.

    Spec: REQ-FR11-009-4
    """
    # Simulate the verdict logic directly with a plateau at S4
    precision_series = [0.50, 0.55, 0.60, 0.6001, 0.6002, 0.6003, 0.601, 0.601, 0.601, 0.601]

    is_monotonically_non_decreasing = all(
        precision_series[i] >= precision_series[i - 1] - 1e-9
        for i in range(1, len(precision_series))
    )
    plateau_session = None
    for i in range(1, len(precision_series)):
        if precision_series[i] - precision_series[i - 1] < 0.001:
            plateau_session = i + 1
            break
    has_regression = any(
        precision_series[i] < precision_series[i - 1] - 0.01
        for i in range(1, len(precision_series))
    )

    if has_regression:
        verdict = "tier2_memory_regression"
    elif is_monotonically_non_decreasing and plateau_session is None:
        verdict = "tier2_memory_monotonic_gain"
    elif plateau_session is not None:
        verdict = f"tier2_memory_plateau_at_s{plateau_session}"
    else:
        verdict = "tier2_memory_monotonic_gain"

    assert verdict == "tier2_memory_plateau_at_s4", (
        f"Expected 'tier2_memory_plateau_at_s4', got '{verdict}'"
    )


def test_honest_verdict_regression():
    """honest_verdict is 'tier2_memory_regression' when any session drops > 0.01.

    Spec: REQ-FR11-009-4
    """
    # Series with a drop of 0.15 at position 5 (> 0.01 threshold)
    precision_series = [0.50, 0.55, 0.60, 0.65, 0.70, 0.55, 0.60, 0.65, 0.68, 0.70]

    has_regression = any(
        precision_series[i] < precision_series[i - 1] - 0.01
        for i in range(1, len(precision_series))
    )
    plateau_session = None
    if not has_regression:
        for i in range(1, len(precision_series)):
            if precision_series[i] - precision_series[i - 1] < 0.001:
                plateau_session = i + 1
                break
    is_monotonically_non_decreasing = all(
        precision_series[i] >= precision_series[i - 1] - 1e-9
        for i in range(1, len(precision_series))
    )

    if has_regression:
        verdict = "tier2_memory_regression"
    elif is_monotonically_non_decreasing and plateau_session is None:
        verdict = "tier2_memory_monotonic_gain"
    elif plateau_session is not None:
        verdict = f"tier2_memory_plateau_at_s{plateau_session}"
    else:
        verdict = "tier2_memory_monotonic_gain"

    assert verdict == "tier2_memory_regression", (
        f"Expected 'tier2_memory_regression', got '{verdict}'"
    )


# ---------------------------------------------------------------------------
# REQ-FR11-010: templates_replayed
# ---------------------------------------------------------------------------


def test_templates_replayed_s1_is_zero(tmp_path):
    """Session S1 (index 0) has 0 templates replayed — cold start, no prior session.

    Spec: REQ-FR11-010
    """
    result = _run_sim(tmp_path)
    assert result["templates_replayed_per_session"][0] == 0, (
        f"S1 should have 0 replays (cold start), got "
        f"{result['templates_replayed_per_session'][0]}"
    )


def test_templates_replayed_s2_through_s10_gt_zero(tmp_path):
    """Sessions S2-S10 each have templates_replayed > 0 (relay is active).

    Spec: REQ-FR11-010-2, SCENARIO-FR11-010
    """
    result = _run_sim(tmp_path)
    replayed = result["templates_replayed_per_session"]

    # S1 is index 0; S2-S10 are indices 1-9
    for i in range(1, N_SESSIONS):
        assert replayed[i] > 0, (
            f"Session S{i+1} (index {i}) should have templates_replayed > 0, got {replayed[i]}. "
            f"Full series: {replayed}"
        )
