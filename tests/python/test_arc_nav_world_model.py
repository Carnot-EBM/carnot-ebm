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


HAZ_CTR = 15  # the charger's centre-marker colour (its offset within the block encodes facing)


def _haz_transitions_facing():
    """Like _haz_transitions but the charger carries a colour-15 centre marker OFFSET to the LEFT of its
    block centre -> it faces LEFT (charges horizontally, to the left)."""
    tr = list(_transitions())
    g0 = np.full((7, 9), WALL, dtype=int)
    g0[3, 1] = AV
    g0[3, 4:7] = HAZ; g0[2, 5] = HAZ; g0[4, 5] = HAZ  # noqa: E702  3x3 charger centred (3,5)
    g0[3, 4] = HAZ_CTR                                  # marker offset LEFT of centre -> faces left
    g1 = np.full((7, 9), WALL, dtype=int)              # avatar REMOVED (death)
    g1[3, 2:5] = HAZ; g1[2, 3] = HAZ; g1[4, 3] = HAZ  # noqa: E702  charger charged left to ~(3,3)
    g1[3, 2] = HAZ_CTR
    for _ in range(3):
        tr.append((g0, 4, g1, 0, 0))
    return tr


def _omni_model():
    """A HazardAwareNavWorldModel with explicit params (the tiny test fixtures are below the fit's blob
    thresholds for the centre marker, so we exercise the is_lethal facing logic directly)."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    return HazardAwareNavWorldModel(
        displacement={1: (-6, 0), 2: (6, 0), 3: (0, -6), 4: (0, 6)},
        avatar_colors=frozenset({AV}), bg_color=WALL, floor_color=FLOOR, wall_colors=frozenset({WALL}),
        goal_color=GOAL, hazard_colors=frozenset({HAZ, HAZ_CTR}), hazard_center_color=HAZ_CTR,
        hazard_axis="row", charge_range=6, lethal_mode="omni")


def test_omni_mode_is_facing_directional():
    """REQ: lethal_mode='omni' (calibrated vs tu93 L3) kills only when the avatar's destination is on a
    charger's FACING line, on the side it faces, within range -- a perpendicular step ONTO that side is
    lethal, landing BEHIND the charger (the side it does not face) is safe, and landing exactly ON the
    charger (collision) is NOT lethal."""
    m = _omni_model()
    # charger 3x3 centred (10,20) with its colour-15 marker offset LEFT -> faces left (charges left).
    def grid_with_avatar(ar, ac):
        g = np.full((25, 40), FLOOR, dtype=int)      # open floor background (no walls to block the charge)
        g[9:12, 19:22] = HAZ; g[10, 19] = HAZ_CTR  # noqa: E702  marker left of centre -> faces left
        g[ar, ac] = AV
        return g
    # avatar to the charger's LEFT, off-row, an UP move ends it ON the charger's row, on the facing side, in range
    assert m.is_lethal(grid_with_avatar(16, 14), 1) is True
    # avatar BEHIND the charger (to its RIGHT, the side it does NOT face), aligned in range -> SAFE
    assert m.is_lethal(grid_with_avatar(16, 26), 1) is False
    # landing exactly ON the charger (collision) -> NOT lethal (it is defeated/passed, not a kill)
    assert m.is_lethal(grid_with_avatar(16, 20), 1) is False


def test_omni_los_uses_per_charger_facing_axis_not_fitted_axis():
    """REQ-ARC-WMTE-5880: in 'omni' mode the wall line-of-sight check must run along the axis the charge
    actually TRAVELS (each charger's OWN facing), not the single fitted self.hazard_axis. A VERTICAL charger
    (faces down its column) with a wall on that column between it and the avatar must NOT be able to charge
    THROUGH the wall -- even when hazard_axis was fitted as 'row'. Before the fix, _charge_unobstructed
    checked the row segment (no wall there), missed the column wall, and flagged the shielded move lethal ->
    over-pruning a genuinely-safe move."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    BG = FLOOR_ = 5
    AV_, WALL_, HZ_, CTR_, GOAL_ = 9, 3, 8, 15, 14
    m = HazardAwareNavWorldModel(
        displacement={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)},
        avatar_colors=frozenset({AV_}), bg_color=BG, floor_color=FLOOR_, wall_colors=frozenset({WALL_}),
        goal_color=GOAL_, hazard_colors=frozenset({HZ_, CTR_}), hazard_center_color=CTR_,
        hazard_axis="row", charge_range=4, lethal_mode="omni", align_tol=1,
    )

    def grid(with_wall):
        g = np.full((8, 10), BG, dtype=int)
        g[2, 5] = HZ_
        g[3, 5] = CTR_  # centre marker one row below the body -> faces DOWN (vertical / column charger)
        if with_wall:
            g[4, 5] = WALL_  # wall on the charging COLUMN, between charger (row 2) and avatar dest (row 5)
        g[6, 5] = AV_
        return g

    # facing is read correctly as vertical/down
    assert m._charger_facing(grid(False), 2, 5) == (1, 0)
    # UP move ends the avatar on the charger's column, in range: lethal WITHOUT the wall, SAFE WITH it
    assert m.is_lethal(grid(False), 1) is True
    assert m.is_lethal(grid(True), 1) is False


def test_fit_prunes_distant_codirectional_decoy_keeps_contiguous_marker():
    """REQ-ARC-WMTE-5882: fit's avatar co-translation admits an INDEPENDENTLY-moving decoy that shares the
    avatar's shift on the aligned subset of moves; the spatial-adjacency prune drops it (it sits elsewhere on
    the grid) while KEEPING a genuine contiguous component (a centre marker inside the body). The discriminator
    is spatial contiguity, not motion -- a decoy and a real marker can both co-shift, but only the marker
    adjoins the body."""
    AV_, CTR_, DECOY_, GOAL_, BG_ = 9, 4, 7, 14, 5
    DIRS = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    H = W = 22
    rng = np.random.default_rng(11)
    top = (2, 2)
    decoy = (15, 0)

    def render(top, decoy, with_decoy):
        g = np.full((H, W), BG_, dtype=int)
        g[18, 18] = GOAL_
        g[top[0]:top[0]+3, top[1]:top[1]+3] = AV_   # 3x3 ring body
        g[top[0]+1, top[1]+1] = CTR_                 # centre marker INSIDE the body (contiguous)
        if with_decoy:
            g[decoy[0], decoy[1]] = DECOY_           # independent mover, elsewhere on the grid
        return g

    def build(with_decoy):
        tr = []
        g = render(top, decoy, with_decoy)
        t, d = top, decoy
        for _ in range(400):
            a = int(rng.integers(1, 5))
            dd = DIRS[a]
            nt = (t[0]+dd[0], t[1]+dd[1])
            if not (0 <= nt[0] and nt[0]+3 <= H and 0 <= nt[1] and nt[1]+3 <= W):
                nt = t
            nd = (d[0], (d[1]+1) % W)
            g2 = render(nt, nd, with_decoy)
            tr.append((g, a, g2, 0, 0))
            g, t, d = g2, nt, nd
        return tr

    m = InducedNavWorldModel.fit(build(with_decoy=True))
    assert AV_ in m.avatar_colors and CTR_ in m.avatar_colors  # contiguous body + marker kept
    assert DECOY_ not in m.avatar_colors                        # distant decoy pruned
    # control: no decoy -> the contiguous {9,4} avatar is still recovered unchanged
    m2 = InducedNavWorldModel.fit(build(with_decoy=False))
    assert AV_ in m2.avatar_colors and CTR_ in m2.avatar_colors


def test_charger_facing_picks_nearest_marker_and_handles_centered():
    """REQ-ARC-WMTE-5881: _charger_facing must read THIS charger's OWN marker (nearest to its blob centre),
    not the row-major-first marker in the +-4 window -- otherwise, with two chargers close together, a
    neighbour's marker inverts this charger's facing. And a marker sitting essentially ON the centre carries
    no directional signal -> return None (unknown), so is_lethal falls back to both-axis rather than a
    spurious definite facing."""
    from carnot.agentic.arc_nav_world_model import HazardAwareNavWorldModel
    BG_, AV_, WALL_, HZ_, CTR_, GOAL_ = 5, 9, 3, 8, 15, 14
    m = HazardAwareNavWorldModel(
        displacement={1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}, avatar_colors=frozenset({AV_}),
        bg_color=BG_, floor_color=BG_, wall_colors=frozenset({WALL_}), goal_color=GOAL_,
        hazard_colors=frozenset({HZ_, CTR_}), hazard_center_color=CTR_, hazard_axis="row",
        charge_range=4, lethal_mode="omni", align_tol=1,
    )
    # Charger A body at (10,10) with its marker BELOW -> faces DOWN. A NEIGHBOUR charger's marker sits at
    # (8,10), which is inside A's +-4 window AND row-major-first. A's facing must still read DOWN (1,0).
    g = np.full((20, 20), BG_, dtype=int)
    g[10, 10] = HZ_
    g[11, 10] = CTR_  # A's own marker, below A -> faces down
    g[8, 12] = HZ_
    g[8, 10] = CTR_   # neighbour marker, row-major-first, would invert to (-1,0) if wrongly picked
    assert m._charger_facing(g, 10, 10) == (1, 0)
    # Centered marker (marker exactly on the blob centroid) -> None (unknown facing)
    g2 = np.full((12, 12), BG_, dtype=int)
    g2[5, 5] = HZ_
    g2[5, 7] = HZ_
    g2[5, 6] = CTR_  # 3-cell blob centroid (5,6); marker at (5,6) -> zero offset
    assert m._charger_facing(g2, 5, 6) is None
