"""Unit tests for the ARC heuristic SELECTION learning (arc_heuristic_select) — the encoded
"when (and when NOT) to use which goal-distance heuristic" rule.

These pin the LEARNED rule itself (derived from the 2026-06-17 8-game A/B), independent of any
live search: no win-state ⇒ BFS only; high per-action cell-impact ⇒ region-count first;
low ⇒ cell-count first; BFS always the complete fallback.
"""
import types

import numpy as np

from carnot.agentic import arc_heuristic_select as sel


def _trans(grid, next_grid):
    return types.SimpleNamespace(grid=np.asarray(grid), next_grid=np.asarray(next_grid))


def test_no_win_state_uses_bfs_only():
    # The 'when NOT to use a heuristic' case: a goal-distance heuristic is meaningless without a
    # target (e.g. a first-ever solve), so the rule must fall back to pure BFS.
    assert sel.recommend_order(step_cells=999, has_target=False) == ["bfs"]
    assert sel.recommend_order(step_cells=0, has_target=False) == ["bfs"]


def test_high_cell_impact_prefers_region_count():
    order = sel.recommend_order(step_cells=70, has_target=True)
    assert order[0] == "region_count"
    assert order[-1] == "bfs"            # BFS is always the complete fallback
    assert set(order) == {"region_count", "cell_count", "bfs"}


def test_low_cell_impact_prefers_cell_count():
    order = sel.recommend_order(step_cells=20, has_target=True)
    assert order[0] == "cell_count"
    assert order[-1] == "bfs"


def test_threshold_boundary():
    # Exactly at the threshold counts as high-impact (>=).
    assert sel.recommend_order(sel.HIGH_IMPACT_CELLS, True)[0] == "region_count"
    assert sel.recommend_order(sel.HIGH_IMPACT_CELLS - 1, True)[0] == "cell_count"


def test_per_action_cell_impact_is_median_changed_cells():
    win = np.zeros((4, 4), dtype=np.int16)
    g1 = win.copy()
    a = g1.copy(); a[0, 0] = 1                    # 1 cell changed
    b = a.copy(); b[1, :] = 2                      # 4 cells changed
    c = b.copy(); c[2, :3] = 3                     # 3 cells changed
    trans = [_trans(g1, a), _trans(a, b), _trans(b, c)]
    # changed-cell counts = [1, 4, 3] -> median 3
    assert sel.per_action_cell_impact(trans) == 3.0


def test_per_action_cell_impact_ignores_noops():
    g = np.zeros((3, 3), dtype=np.int16)
    moved = g.copy(); moved[0, 0] = 5
    trans = [_trans(g, g), _trans(g, moved)]      # first is a no-op (0 changed) -> ignored
    assert sel.per_action_cell_impact(trans) == 1.0


def test_factory_returns_callables_and_none_for_bfs():
    win = np.zeros((5, 5), dtype=np.int16)
    grid = win.copy(); grid[0, 0] = 1; grid[4, 4] = 1   # two separate wrong regions
    region = sel.factory("region_count", win)
    cell = sel.factory("cell_count", win)
    assert sel.factory("bfs", win) is None
    assert region(grid) == 2.0          # two 8-connected wrong regions
    assert cell(grid) == 2.0            # two differing cells
    assert region(win) == 0.0 and cell(win) == 0.0


def test_factory_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        sel.factory("does_not_exist", np.zeros((2, 2)))
