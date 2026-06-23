"""Tests for the goal-ENERGY wiring (2026-06-23, closes GAP-ARCH-GOAL-NOT-VERIFIED).

induce_goal_energy is the GRADED counterpart of induce_goal_predicate; plan_in_model(goal_energy=...) makes
the in-model planner BEST-FIRST (descend toward the induced goal) instead of blind FIFO BFS -> fewer
nodes-to-win (the action-efficiency win). The energy is induced per-game from the agent's OWN observed
win/non-win states; an ablation control + a silent-failure guard are mandatory in the live wiring.
"""

import numpy as np

from carnot.agentic.arc_agi3_goal_induction import induce_goal_energy
from carnot.agentic.arc_agi3_world_model import objects
from carnot.agentic.arc_executable_world_model import plan_in_model


def _grid(positions):
    g = np.zeros((9, 9), dtype=int)
    for i, (y, x) in enumerate(positions):
        g[y, x] = i + 1
    return g


_POS = [(1, 1), (1, 7), (4, 4), (7, 1), (7, 7), (2, 5)]


def test_induce_goal_energy_is_graded_and_zero_at_win():
    # goal = reduce objects to <= max_win_objs (2); energy is the violation magnitude
    ge = induce_goal_energy([_grid(_POS[:2]), _grid(_POS[:1]), _grid([])], [_grid(_POS), _grid(_POS[:4])])
    assert ge is not None
    assert ge(_grid(_POS[:2])) == 0.0          # win state (2 objects) -> satisfied
    assert ge(_grid(_POS)) > ge(_grid(_POS[:4]))  # 6 objects farther than 4 objects (graded, monotone)
    assert ge(_grid(_POS)) == 4.0              # 6 objects, ceiling 2 -> violation 4


def test_induce_goal_energy_needs_two_wins():
    assert induce_goal_energy([_grid(_POS[:1])], [_grid(_POS)]) is None  # <2 win examples


def _remove_click_engine(counter, tag):
    def engine(grid, action, data):
        counter[tag] += 1
        g = grid.copy()
        if action == 6 and data is not None:  # click removes the object at (x, y)
            g[int(data["y"]), int(data["x"])] = 0
        return g
    return engine


def test_plan_in_model_goal_energy_reaches_win_in_fewer_nodes():
    start = _grid(_POS)
    K = 2

    def is_win(g):
        return len(objects(g)) <= K

    ge = induce_goal_energy([_grid(_POS[:2]), _grid(_POS[:1]), _grid([])], [_grid(_POS), _grid(_POS[:3])])
    counts = {"bfs": 0, "ge": 0}
    p_bfs = plan_in_model(_remove_click_engine(counts, "bfs"), is_win, start, max_nodes=20000, max_depth=10)
    p_ge = plan_in_model(
        _remove_click_engine(counts, "ge"), is_win, start, max_nodes=20000, max_depth=10, goal_energy=ge
    )
    assert p_bfs is not None and p_ge is not None          # both reach the win
    assert len(p_ge) == len(p_bfs)                         # same plan length (4 removals)
    assert counts["ge"] < counts["bfs"]                    # goal-energy explores fewer nodes
    assert counts["ge"] * 2 < counts["bfs"]                # materially fewer (observed ~6x)


def test_plan_in_model_backward_compatible_without_goal_energy():
    # goal_energy=None must keep the exact original FIFO BFS behaviour
    start = _grid(_POS)

    def is_win(g):
        return len(objects(g)) <= 2

    counts = {"bfs": 0}
    p = plan_in_model(_remove_click_engine(counts, "bfs"), is_win, start, max_nodes=20000, max_depth=10)
    assert p is not None and len(p) == 4
