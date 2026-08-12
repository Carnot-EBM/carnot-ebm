"""REQ-ARC-WMTE-6257: a constant-False goal predicate must not score as perfect.

WHY. `score_goal_predicate_consistency`'s `accuracy` is a SPECIFICITY score. Held-out
transitions from `collect_transitions` contain no level-ups, so every row is a non-win and a
predicate returning False for everything is 100% correct. Measured 2026-08-12: 14 of 21
stored predicates scored a perfect 1.0 that way while never firing on a real win, and
`plan_in_model` terminates on this predicate -- 10 of them made planning impossible and 4
produced hollow plans that ended on an in-model false win.

Covers SCENARIO-ARC-WMTE-6257-CONSTANT-FALSE-MUST-NOT-SCORE-PERFECT.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from carnot.agentic.arc_executable_world_model import score_goal_predicate_consistency


@dataclass
class _T:
    grid: np.ndarray
    action: int
    data: dict | None
    next_grid: np.ndarray
    level_before: int = 0
    level_after: int = 0


def _non_win_transitions(n: int = 5) -> list[_T]:
    """Held-out data as the collector really produces it: no level-ups anywhere."""
    rng = np.random.default_rng(6257)
    out = []
    for _ in range(n):
        g = rng.integers(0, 4, size=(4, 4))
        out.append(_T(grid=g, action=1, data=None, next_grid=g + 1))
    return out


def _win_grid() -> np.ndarray:
    return np.full((4, 4), 9)


def test_constant_false_scores_perfect_specificity_but_is_flagged_degenerate() -> None:
    # The exact failure mode found in the store: looks perfect, decides nothing.
    out = score_goal_predicate_consistency(
        lambda g: False, _non_win_transitions(), win_grids=[_win_grid()]
    )
    assert out.accuracy == 1.0, "specificity alone still rates it perfect -- that is the point"
    assert out.sensitivity_win_grids_tested == 1
    assert out.sensitivity_win_grids_fired == 0
    assert out.is_degenerate_constant_false is True


def test_a_discriminating_predicate_is_not_flagged() -> None:
    win = _win_grid()
    out = score_goal_predicate_consistency(
        lambda g: bool(np.array_equal(np.asarray(g), win)),
        _non_win_transitions(),
        win_grids=[win],
    )
    assert out.accuracy == 1.0
    assert out.sensitivity_win_grids_fired == 1
    assert out.is_degenerate_constant_false is False


def test_no_win_grids_leaves_sensitivity_UNMEASURED_not_passing() -> None:
    # "Nobody checked" must never look like "checked and fine". This is the fail-safe
    # direction: absent evidence, the field is None rather than False.
    out = score_goal_predicate_consistency(lambda g: False, _non_win_transitions())
    assert out.sensitivity_win_grids_tested == 0
    assert out.is_degenerate_constant_false is None


def test_a_predicate_that_raises_counts_as_not_firing_and_fails_specificity_instead() -> None:
    """A crashing predicate is broken, but it is NOT the looks-perfect degeneracy.

    Written expecting `is_degenerate_constant_false is True`; the code returned False and the
    code was right. A predicate that raises also raises inside the specificity loop, so it
    scores 0.0 there and is already failing loudly. The degenerate flag is reserved for the
    silent case -- high specificity, no fire -- which is the one nothing else catches.
    """

    def boom(_g):
        raise RuntimeError("predicate crashed on the win state")

    out = score_goal_predicate_consistency(boom, _non_win_transitions(), win_grids=[_win_grid()])
    assert out.sensitivity_win_grids_tested == 1
    assert out.sensitivity_win_grids_fired == 0
    assert out.accuracy == 0.0, "a raising predicate is caught by specificity"
    assert out.is_degenerate_constant_false is False


def test_constant_true_is_caught_by_specificity_not_by_the_new_flag() -> None:
    # The opposite degeneracy. It fires on the win, so the new flag is False; specificity
    # is what fails it. Both sides are needed and neither is redundant.
    out = score_goal_predicate_consistency(
        lambda g: True, _non_win_transitions(), win_grids=[_win_grid()]
    )
    assert out.accuracy < 0.5
    assert out.sensitivity_win_grids_fired == 1
    assert out.is_degenerate_constant_false is False


def test_existing_callers_are_unaffected_when_win_grids_is_omitted() -> None:
    # Backward compatibility: the new parameter is optional and the old fields keep their
    # meaning, so no existing call site changes behaviour.
    out = score_goal_predicate_consistency(lambda g: False, _non_win_transitions())
    assert out.accuracy == 1.0
    assert out.n == 5
    assert out.n_real_levelups == 0
