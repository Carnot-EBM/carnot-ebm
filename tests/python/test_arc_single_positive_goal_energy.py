"""Tests for the single-WIN-exemplar goal-energy operator (GAP-4890).

The within-game L2->L3 deepening unblock: at a game's solved frontier the agent has only ONE
level-completion exemplar, below induce_goal_energy's >=2 floor. induce_goal_energy_single_positive
fires from ONE win grid IFF the win is strictly separated from every negative (anti-mis-induction),
and returns None when nothing separates. Spec coverage: GAP-4890 (ops/verifier_gaps.md).
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_agi3_goal_induction import (
    induce_goal_energy,
    induce_goal_energy_richer,
    induce_goal_energy_single_positive,
)


def _grid_with_objects(n: int, color: int = 1, size: int = 7) -> np.ndarray:
    """Background-0 grid with n separated single-cell objects (4-neighbour isolated)."""
    g = np.zeros((size, size), dtype=int)
    placed, r, c = 0, 0, 0
    while placed < n:
        g[r, c] = color
        placed += 1
        c += 2  # gap of 1 keeps cells in separate connected components
        if c >= size:
            c = 0
            r += 2
            if r >= size:
                break
    return g


def test_single_positive_fires_where_two_win_floor_blocks():
    """GAP-4890 core: 1 win grid -> induce_goal_energy returns None, single-positive FIRES and separates."""
    win = _grid_with_objects(1)            # the lone level-completion exemplar (1 object)
    negs = [_grid_with_objects(3), _grid_with_objects(4)]  # self-played non-win states (more objects)

    # the existing operator is blocked by the >=2 win floor (the exact stall this closes)
    assert induce_goal_energy([win], negs) is None

    energy = induce_goal_energy_single_positive(win, negs)
    assert energy is not None, "single-positive operator must fire from one separated win"
    # H1: goal = reduce objects to the win's count -> 0 on the win, >0 on every negative
    assert energy(win) == 0.0
    assert all(energy(g) > 0.0 for g in negs), "induced energy must separate win from non-wins"


def test_single_positive_returns_none_when_no_feature_separates():
    """Anti-mis-induction guard: if the lone win is indistinguishable from the negatives, return None."""
    win = _grid_with_objects(3, color=1)
    negs = [_grid_with_objects(3, color=1)]  # identical object-count AND colours
    assert induce_goal_energy_single_positive(win, negs) is None


def test_single_positive_colour_reduction_hypothesis():
    """H3: a win with strictly fewer unique colours than every negative fires a colour-reduction energy."""
    win = np.zeros((5, 5), dtype=int)
    win[0, 0] = 1  # colours {0,1}
    neg = np.zeros((5, 5), dtype=int)
    neg[0, 0] = 1
    neg[0, 2] = 2
    neg[2, 0] = 3  # colours {0,1,2,3}, same-ish object structure but more colours
    energy = induce_goal_energy_single_positive(win, [neg])
    assert energy is not None
    assert energy(win) == 0.0
    assert energy(neg) > 0.0


def test_single_positive_handles_empty_inputs():
    """Defensive: no negatives or no win -> None (cannot contrast)."""
    assert induce_goal_energy_single_positive(None, [_grid_with_objects(2)]) is None
    assert induce_goal_energy_single_positive(_grid_with_objects(1), []) is None


# --- GAP-4891 richer goal-feature family --------------------------------------------------------

def test_richer_fires_on_fill_where_counts_are_blind():
    """GAP-4891 core: a region-FILL goal -- win has more non-background cells than non-wins, but the
    SAME object/colour COUNTS. The count-only operator returns None; the richer operator fires on the
    nonbg_cells (fill) feature and separates."""
    win = np.zeros((6, 6), dtype=int)
    win[0:4, 0:4] = 1  # 16 filled cells, 1 connected object, colours {0,1}
    nonwin = np.zeros((6, 6), dtype=int)
    nonwin[0, 0:5] = 1  # 5 filled cells, 1 connected object, colours {0,1}

    # counts are identical (1 object, 2 colours) -> the GAP-4890 count operator cannot separate
    assert induce_goal_energy_single_positive(win, [nonwin]) is None

    energy = induce_goal_energy_richer(win, [nonwin])
    assert energy is not None, "richer operator must fire on a fill goal the counts are blind to"
    assert energy(win) == 0.0
    assert energy(nonwin) > 0.0


def test_richer_returns_none_when_truly_indistinguishable():
    """If NO feature (count or value/fill/spatial) separates the lone win from the negatives -> None."""
    win = np.zeros((6, 6), dtype=int)
    win[0:2, 0:2] = 1
    nonwin = np.zeros((6, 6), dtype=int)
    nonwin[0:2, 0:2] = 1  # identical grid -> every feature identical
    assert induce_goal_energy_richer(win, [nonwin]) is None


def test_richer_handles_empty_inputs():
    assert induce_goal_energy_richer(None, [np.zeros((4, 4), dtype=int)]) is None
    assert induce_goal_energy_richer(np.zeros((4, 4), dtype=int), []) is None
