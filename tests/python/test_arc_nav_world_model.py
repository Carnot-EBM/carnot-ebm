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


# --- hazard-aware model: a horizontal line-charger HAZARD (colour 8) on the avatar's row ---------------
HAZ = 8


def _haz_transitions():
    """Move + block + a LETHAL transition: the avatar approaches a colour-8 charger along its row; the
    charger CHARGES (moves toward the avatar) and REMOVES it (avatar absent in g1)."""
    tr = list(_transitions())  # the nav move/block/levelup transitions (no hazard present)
    # lethal: avatar at (3,1) on row 3, charger block at (3,5); avatar moves right (toward it); charger
    # charges left to intercept and the avatar is REMOVED.
    g0 = np.full((7, 9), WALL, dtype=int)
    g0[3, 1] = AV
    g0[3, 4:7] = HAZ; g0[2, 5] = HAZ; g0[4, 5] = HAZ  # noqa: E702 — a colour-8 charger blob around (3,5)
    g1 = np.full((7, 9), WALL, dtype=int)                  # avatar REMOVED (death)
    g1[3, 2:5] = HAZ; g1[2, 3] = HAZ; g1[4, 3] = HAZ  # noqa: E702 — charger CHARGED left (moved) to ~(3,3)
    for _ in range(3):
        tr.append((g0, 4, g1, 0, 0))                       # repeat so the learner has a clear signal
    return tr


def test_hazard_aware_learns_line_charger():
    """REQ: the hazard-aware model learns the charger colour, axis (row), and a positive charge range."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    m = HazardAwareNavWorldModel.fit(_haz_transitions(), goal_color=GOAL)
    assert HAZ in m.hazard_colors          # learned the charger colour (the object that MOVED at death)
    assert m.hazard_axis == "row"          # horizontal charger
    assert m.charge_range > 0
    assert m.goal_color == GOAL            # inherited / level-invariant goal


def test_hazard_aware_engine_predicts_avatar_removal_on_lethal_move():
    """REQ: a move into the charger's range is predicted as avatar-REMOVAL (a dead-end the planner avoids);
    a move AWAY from the charger is a normal safe nav move."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    m = HazardAwareNavWorldModel.fit(_haz_transitions(), goal_color=GOAL)
    # avatar adjacent to the charger on the same row, moving toward it -> lethal -> avatar erased
    g = np.full((7, 9), WALL, dtype=int)
    g[3, 1] = AV
    g[3, 4:7] = HAZ; g[2, 5] = HAZ; g[4, 5] = HAZ  # noqa: E702
    assert m.is_lethal(g, 4) is True
    out = m.engine(g, 4)
    assert not np.any(out == AV)           # the avatar was removed (dead-end grid)
    # moving the other way (away from the charger) is NOT lethal
    assert m.is_lethal(g, 3) is False


def test_hazard_fit_excludes_the_door_colour():
    """REQ: the door colour (passable, everywhere) must NOT be learned as a hazard, even though door blobs
    sit near the avatar -- only the object that MOVES at death is the charger."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    m = InducedNavWorldModel.fit(_transitions())
    assert m.door_color == DOOR            # the base model captures the passable door colour
    h = HazardAwareNavWorldModel.fit(_haz_transitions(), goal_color=GOAL)
    assert DOOR not in h.hazard_colors     # the door is excluded from hazard candidates


def test_enter_mode_catches_perpendicular_step_on_that_toward_misses():
    """REQ: lethal_mode='enter' flags a perpendicular step ONTO the charge line (tu93 L3 vertical chargers),
    which lethal_mode='toward' (along-axis approach only) does not."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    base = HazardAwareNavWorldModel.fit(_haz_transitions(), goal_color=GOAL)  # axis=row, range>0
    # avatar 3 rows BELOW the charger's row (genuinely OFF the row: 3 > align_tol 2), same column band; an UP
    # move (step 2) steps perpendicularly ONTO the charger's row band -> 'enter' lethal, 'toward' safe.
    g = np.full((7, 9), WALL, dtype=int)
    g[5, 5] = AV                            # row 5, charger centred row 2 -> 3 rows off-line before the move
    g[1, 5] = HAZ; g[2, 4] = HAZ; g[2, 5] = HAZ  # noqa: E702
    g[2, 6] = HAZ; g[3, 5] = HAZ            # charger blob centred ~(2,5)
    toward = HazardAwareNavWorldModel.fit(_haz_transitions(), goal_color=GOAL, lethal_mode="toward")
    enter = HazardAwareNavWorldModel.fit(_haz_transitions(), goal_color=GOAL, lethal_mode="enter")
    # action 1 = up = perpendicular step onto the charger's row band
    assert toward.is_lethal(g, 1) is False
    assert enter.is_lethal(g, 1) is True
