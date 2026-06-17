"""Unit tests for the move-distance-aware ARC goal heuristic
`misplaced_region_distance` (python/carnot/agentic/arc_graph_explore.py).

Rationale (2026-06-17): a raw cell-count goal-distance over-estimates MOVE-distance in
ARC-AGI-3 games where one action changes many cells, so A* goes greedy and finds suboptimal
paths (proven by a heuristic_weight sweep on r11l). The misplaced-REGION count (connected
components of `grid != win`) is move-aligned — each action typically fixes one region, so it
drops ~1 per move, giving both the right scale and a real gradient. These tests pin the
heuristic's contract: zero at the win, region-counting semantics, 8- vs 4-connectivity, and
finite-float output.
"""
import numpy as np
import pytest

from carnot.agentic.arc_graph_explore import misplaced_region_distance


def test_zero_distance_at_the_win():
    # At the win-state there are no 'wrong regions' -> distance must be exactly 0.0.
    win = np.array([[0, 1], [2, 3]], dtype=np.int16)
    gd = misplaced_region_distance(win)
    assert gd(win) == 0.0
    assert isinstance(gd(win), float)


def test_counts_separate_wrong_regions():
    # Two spatially-separated mismatched cells = two distinct wrong regions (move-aligned).
    win = np.zeros((5, 5), dtype=np.int16)
    grid = win.copy()
    grid[0, 0] = 7          # region A (top-left)
    grid[4, 4] = 7          # region B (bottom-right) — not adjacent to A
    gd = misplaced_region_distance(win, connectivity=8)
    assert gd(grid) == 2.0


def test_8_connectivity_merges_diagonal_mismatches():
    # A diagonal pair of mismatched cells is ONE region under 8-connectivity (the empirically
    # better setting — actions group their changes), but TWO under 4-connectivity.
    win = np.zeros((3, 3), dtype=np.int16)
    grid = win.copy()
    grid[0, 0] = 5
    grid[1, 1] = 5          # diagonally adjacent to (0,0)
    assert misplaced_region_distance(win, connectivity=8)(grid) == 1.0
    assert misplaced_region_distance(win, connectivity=4)(grid) == 2.0


def test_returns_finite_float_for_arbitrary_grid():
    # Contract: a heuristic must always return a finite float (never None / NaN) so the
    # search's priority queue stays well-ordered.
    win = np.zeros((8, 8), dtype=np.int16)
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 4, size=(8, 8)).astype(np.int16)
    val = misplaced_region_distance(win)(grid)
    assert isinstance(val, float)
    assert np.isfinite(val)
    assert val >= 0.0


def test_monotone_under_fixing_a_region():
    # Fixing one wrong region (setting it back to the win value) must NOT increase the
    # distance — the property that makes A* converge toward the win.
    win = np.zeros((6, 6), dtype=np.int16)
    grid = win.copy()
    grid[0, 0] = 1
    grid[0, 5] = 1
    grid[5, 0] = 1
    gd = misplaced_region_distance(win, connectivity=8)
    before = gd(grid)
    grid[0, 0] = 0          # fix one region
    after = gd(grid)
    assert after <= before
    assert after == before - 1.0
