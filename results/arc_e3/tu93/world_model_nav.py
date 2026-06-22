"""tu93 grid-level executable world model — HAND-INDUCED by the outer loop (Claude as proposer,
2026-06-22) as the program-generalization POSITIVE CONTROL.

(Distinct from this dir's world_model.py, which is a tiny BranchState L5-parity unit fixture, not a
loadable engine. This file is the full grid engine(grid,action,data)->grid + is_level_complete(grid).)

tu93 is a clean 4-direction maze navigation game (registry mechanic_class: graph_explore). Reverse-
engineered from offline transitions (scripts/experiments/experiment_program_gen.py geometry probe):

  * Logical cell size = 1 (frames already at game resolution, 64x64).
  * The AVATAR is a 3x3 block: 8 cells of colour 9 around a colour-4 centre marker.
  * Each keyboard ACTION translates the avatar by exactly 6 pixels (one maze pitch = 3-wide room +
    3-wide wall): ACTION1=up, ACTION2=down, ACTION3=left, ACTION4=right.
  * Walls are colour 2; open room cells are colour 0; background is colour 5; GOAL is colour 14. A move
    SUCCEEDS only if the 6px destination footprint and the mid-gap it sweeps are free of wall colour 2
    and in-bounds; otherwise the avatar stays put (wall collision).
  * The level completes when the avatar reaches the colour-14 goal.

This transition + goal is LEVEL-INVARIANT by construction (maze layout/goal change per level, the move
dynamics + win condition do not) -- exactly the property the Executable-World-Models leader
(arXiv:2605.05138) exploits to DEEPEN by planning in imagination. The move counter (a strip the env
decrements each step) is deliberately NOT modelled: it is irrelevant to navigation and ticking it would
make every state unique and defeat BFS de-duplication. Scored via WorldModelVerifier changed-cell-recall
(the granularity-matched gate), not full-grid exact match.
"""
from __future__ import annotations

import numpy as np

_DIRS = {1: (-6, 0), 2: (6, 0), 3: (0, -6), 4: (0, 6)}  # up, down, left, right; step = one maze pitch
# Empirically derived blocking rule (experiment_program_gen geometry probe, 120 transitions):
#   colour 5 = impassable WALL/background; colour 2 = passable DOORWAY between rooms; colour 0 = open room.
#   A move is ALLOWED iff the 3x3 mid-gap the avatar sweeps through is the colour-2 door (no colour-5);
#   BLOCKED iff the mid-gap contains the colour-5 wall (avatar stays put, only the env's move counter ticks).
WALL = 5
DOOR = 2
OPEN = 0
GOAL = 14
AVATAR = 9
CENTER = 4


def _avatar_bbox(grid):
    """Top-left (r, c) of the avatar's 3x3 block, located by its colour-4 centre inside a 9-ring."""
    g = np.asarray(grid)
    ys, xs = np.where(g == CENTER)
    for y, x in zip(ys, xs):
        if 1 <= y < g.shape[0] - 1 and 1 <= x < g.shape[1] - 1:
            block = g[y - 1:y + 2, x - 1:x + 2]
            if np.sum(block == AVATAR) >= 6:  # robust to a corner being overwritten
                return y - 1, x - 1
    ys9, xs9 = np.where(g == AVATAR)  # fallback: centroid of the 9-block
    if ys9.size:
        return int(round(ys9.mean())) - 1, int(round(xs9.mean())) - 1
    return None


def _goal_cells(grid):
    g = np.asarray(grid)
    ys, xs = np.where(g == GOAL)
    return list(zip(ys.tolist(), xs.tolist()))


def engine(grid, action, data=None):
    """Predict the next tu93 logical grid for one action. Pure function of the grid (navigation
    subspace only; the move counter is intentionally not modelled)."""
    g = np.asarray(grid).copy()
    a = int(action)
    if a not in _DIRS:
        return g
    bb = _avatar_bbox(g)
    if bb is None:
        return g
    r, c = bb
    dy, dx = _DIRS[a]
    nr, nc = r + dy, c + dx
    H, W = g.shape
    if nr < 0 or nc < 0 or nr + 2 >= H or nc + 2 >= W:  # destination out of bounds
        return g
    my, mx = r + dy // 2, c + dx // 2                   # the 3x3 mid-gap the avatar sweeps through
    if np.any(g[my:my + 3, mx:mx + 3] == WALL):         # colour-5 wall in the gap -> blocked, avatar stays
        return g
    stamp = g[r:r + 3, c:c + 3].copy()
    g[r:r + 3, c:c + 3] = OPEN          # clear old footprint to corridor
    # draw avatar at destination -- if the destination IS the colour-14 goal, the avatar COVERS it
    # (goal cells disappear), which is exactly how is_level_complete detects the win.
    g[nr:nr + 3, nc:nc + 3] = stamp
    return g


def is_level_complete(grid):
    """True when the avatar has reached (and thus COVERED) the colour-14 goal: an avatar is present and
    no goal cells remain. The avatar's 3x3 footprint exactly overlays the 3x3 goal on the 6px maze grid,
    so a successful move onto the goal removes every colour-14 cell."""
    g = np.asarray(grid)
    if _avatar_bbox(g) is None:
        return False
    return not bool(np.any(g == GOAL))


def transition_fixture():
    """Self-test: avatar at top-left (7,8) moves DOWN (action 2) by 6px into open corridor."""
    g = np.full((30, 20), 5, dtype=int)
    g[7:16, 7:12] = 0                                  # open corridor column
    g[7, 8] = g[7, 9] = g[7, 10] = 9
    g[8, 8] = g[8, 10] = 9
    g[9, 8] = g[9, 9] = g[9, 10] = 9
    g[8, 9] = 4                                        # avatar centre at (8,9) -> top-left (7,8)
    out = engine(g, 2)
    bb = _avatar_bbox(out)
    return {"transition": "tu93:nav:down6", "new_top_left": bb, "expected": (13, 8),
            "passed": bb == (13, 8)}
