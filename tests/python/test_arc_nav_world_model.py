"""Unit tests for the auto-fitting nav world model (carnot.agentic.arc_nav_world_model).

Covers the re-induction primitive used by scripts/experiments/experiment_reinduction.py: the model must
LEARN {avatar colours, per-action displacement, wall colour, floor colour, goal colour} FROM TRANSITIONS
alone (no hardcoding) and then drive plan_in_model via engine + is_level_complete.

Traceability: program-generalization / mechanic-conditioned re-induction work
(docs/research-notes/program-generalization-first-swing-2026-06-22.md).
"""
from __future__ import annotations

import numpy as np

from carnot.agentic.arc_nav_world_model import InducedNavWorldModel

# A fully-known ground-truth nav mechanic (independent of the fitter):
#   colour 5 = wall/background, 2 = door, 0 = floor, 4 = avatar (1x1), 14 = goal. step = 2.
WALL, DOOR, FLOOR, AV, GOAL = 5, 2, 0, 4, 14
STEP = 2
_DIRS = {1: (-STEP, 0), 2: (STEP, 0), 3: (0, -STEP), 4: (0, STEP)}


def _grid(av_rc, *, mid=None, mid_color=None, dest=None, dest_color=None):
    """A 7x7 wall field with a 1x1 avatar, optionally a coloured mid-gap cell and a coloured dest cell."""
    g = np.full((7, 7), WALL, dtype=int)
    g[av_rc] = AV
    if mid is not None:
        g[mid] = mid_color
    if dest is not None:
        g[dest] = dest_color
    return g


def _mk(av, action, outcome):
    """Build a (g0, action, g1, level_before, level_after) transition with a known outcome for `action`
    from avatar cell `av`: 'move' (door ahead -> avatar advances), 'block' (wall ahead -> stays),
    'levelup' (goal ahead -> avatar covers it, level 0->1)."""
    dy, dx = _DIRS[action]
    r, c = av
    mid = (r + dy // 2, c + dx // 2)
    dest = (r + dy, c + dx)
    if outcome == "block":
        g0 = _grid(av, mid=mid, mid_color=WALL)
        return (g0, action, g0.copy(), 0, 0)
    dest_color = GOAL if outcome == "levelup" else FLOOR
    g0 = _grid(av, mid=mid, mid_color=DOOR, dest=dest, dest_color=dest_color)
    g1 = _grid(dest, mid=mid, mid_color=DOOR)   # avatar now at dest; old cell becomes wall-field floor below
    g1[av] = FLOOR
    la = 1 if outcome == "levelup" else 0
    return (g0, action, g1, 0, la)


def _transitions():
    """Exercise every direction with BOTH a move and a wall-block, plus one goal level-up (so goal colour
    is learned from the level-up signal, not a heuristic fallback)."""
    tr = []
    for action in (1, 2, 3, 4):
        tr.append(_mk((3, 3), action, "move"))
        tr.append(_mk((3, 3), action, "block"))
    tr.append(_mk((3, 3), 4, "move"))                 # extra moves so displacement mode is unambiguous
    tr.append(_mk((3, 1), 4, "move"))
    tr.append(_mk((1, 3), 2, "move"))
    tr.append(_mk((3, 3), 4, "levelup"))              # avatar covers the goal -> level 0->1
    return tr


def test_fit_recovers_mechanic_from_transitions():
    """REQ: the model learns avatar/wall/floor/goal/displacement purely from data (no hardcoding)."""
    m = InducedNavWorldModel.fit(_transitions())
    assert AV in m.avatar_colors
    assert m.floor_color == FLOOR
    assert WALL in m.wall_colors
    assert m.goal_color == GOAL                       # learned from the level-up transition
    assert m.displacement.get(4) == (0, STEP)
    assert m.displacement.get(3) == (0, -STEP)
    assert m.displacement.get(1) == (-STEP, 0)
    assert m.displacement.get(2) == (STEP, 0)


def test_engine_moves_and_blocks_correctly():
    """REQ: the induced engine reproduces ground-truth move/block decisions."""
    m = InducedNavWorldModel.fit(_transitions())
    # door to the right of (3,3): move advances the avatar to (3,5)
    g = _grid((3, 3), mid=(3, 4), mid_color=DOOR, dest=(3, 5), dest_color=FLOOR)
    out = m.engine(g, 4)
    ys, xs = np.where(out == AV)
    assert (int(ys[0]), int(xs[0])) == (3, 5)
    # wall to the right -> blocked, avatar stays at (3,3)
    gb = _grid((3, 3), mid=(3, 4), mid_color=WALL)
    outb = m.engine(gb, 4)
    ys2, xs2 = np.where(outb == AV)
    assert (int(ys2[0]), int(xs2[0])) == (3, 3)


def test_is_level_complete_on_goal_coverage():
    """REQ: level complete exactly when the avatar covers the goal colour."""
    m = InducedNavWorldModel.fit(_transitions())
    with_goal = _grid((3, 3), dest=(3, 5), dest_color=GOAL)
    assert m.is_level_complete(with_goal) is False     # goal still visible
    covered = _grid((3, 5))                             # avatar present, no goal cell remains
    assert m.is_level_complete(covered) is True


def test_fit_does_not_fabricate_on_static_input():
    """REQ: degenerate input must not crash or invent a confident mechanic."""
    static = [(np.full((7, 7), WALL), 1, np.full((7, 7), WALL), 0, 0)]
    m = InducedNavWorldModel.fit(static)
    assert m.displacement == {}
    g = np.full((7, 7), WALL)
    assert np.array_equal(m.engine(g, 4), g)           # safe identity, no motion invented
