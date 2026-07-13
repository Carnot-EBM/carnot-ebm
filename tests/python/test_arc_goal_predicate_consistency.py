"""Tests for arc_executable_world_model.score_goal_predicate_consistency.

Spec refs: REQ-ARC-WMTE-5593, SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR,
SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    score_goal_predicate_consistency,
)


def _grid(value: int = 0) -> np.ndarray:
    grid = np.zeros((3, 3), dtype=np.int16)
    grid[0, 0] = value
    return grid


def test_req_arc_wmte_5593_perfect_predictor_scores_full_accuracy() -> None:
    """SCENARIO-ARC-WMTE-5593-CORRECT-PREDICTOR: a predictor matching real level-ups
    scores 1.0, with n_real_levelups/n_real_noops correctly counted.

    Both transitions share the SAME level boundary (0 -> 1): is_level_complete is a
    per-level-boundary predicate (freshly re-induced after each level-up in the real
    pipeline), so a fair consistency check stays within one boundary's transitions
    rather than spanning multiple level transitions with different induced predicates.
    """

    def is_won(grid: np.ndarray) -> bool:
        return bool(grid[0, 0] == 9)

    transitions = [
        Transition(_grid(0), 1, None, _grid(0), 0, 0),  # no-op
        Transition(_grid(0), 1, None, _grid(9), 0, 1),  # real level-up
    ]

    result = score_goal_predicate_consistency(is_won, transitions)

    assert result.accuracy == 1.0
    assert result.n_correct == 2
    assert result.n == 2
    assert result.n_real_levelups == 1
    assert result.n_real_noops == 1
    assert result.mismatches == []


def test_req_arc_wmte_5593_broken_predictor_caught_not_silently_perfect() -> None:
    """SCENARIO-ARC-WMTE-5593-BROKEN-PREDICTOR-CAUGHT: a predictor that never claims a win
    is caught missing the real level-up, not silently scored as trustworthy."""

    def never_won(grid: np.ndarray) -> bool:
        return False

    transitions = [
        Transition(_grid(0), 1, None, _grid(0), 0, 0),
        Transition(_grid(0), 1, None, _grid(9), 0, 1),  # real level-up, missed
    ]

    result = score_goal_predicate_consistency(never_won, transitions)

    assert result.accuracy == 0.5
    assert result.n_correct == 1
    assert len(result.mismatches) == 1
    assert result.mismatches[0]["real_levelup"] is True
    assert result.mismatches[0]["claimed"] is False


def test_req_arc_wmte_5593_overclaiming_predictor_also_caught() -> None:
    """A predictor that claims EVERY grid is a win is caught false-positiving on no-ops,
    not just missed level-ups -- both directions of miscalibration are detected."""

    def always_won(grid: np.ndarray) -> bool:
        return True

    transitions = [
        Transition(_grid(0), 1, None, _grid(0), 0, 0),  # no-op, falsely claimed as won
        Transition(_grid(0), 1, None, _grid(9), 0, 1),  # real level-up, correctly claimed
    ]

    result = score_goal_predicate_consistency(always_won, transitions)

    assert result.accuracy == 0.5
    assert result.mismatches[0]["real_levelup"] is False
    assert result.mismatches[0]["claimed"] is True


def test_req_arc_wmte_5593_crashing_predictor_counts_as_a_miss_not_a_crash() -> None:
    """A predicate that raises is treated as a claimed=False miss, never propagates the
    exception -- a crashing goal hypothesis must not take down the consistency check."""

    def crashes(grid: np.ndarray) -> bool:
        raise ValueError("bad predicate")

    transitions = [Transition(_grid(0), 1, None, _grid(9), 0, 1)]

    result = score_goal_predicate_consistency(crashes, transitions)

    assert result.n_correct == 0
    assert result.mismatches[0]["error"] is not None


def test_req_arc_wmte_5593_empty_transitions_returns_zero_not_divide_by_zero() -> None:
    """An empty transition list returns a well-formed zero result, no ZeroDivisionError."""

    result = score_goal_predicate_consistency(lambda grid: True, [])

    assert result.n == 0
    assert result.accuracy == 0.0
