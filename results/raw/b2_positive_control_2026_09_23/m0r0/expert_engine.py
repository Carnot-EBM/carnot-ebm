"""Grid-only expert engine for m0r0 level 1 (index 0), written from the public game source.

Uses only (grid, action, data) and constants read from environment_files/m0r0/492f87ba/m0r0.py:
  - Level 1 grid_size (11, 11) -> scale 64//11 = 5, offset (64-55)//2 = 4.
  - Pieces are 1x1 sprites of color 10; floor is BACKGROUND_COLOR 5; walls render as 11/12.
  - Two pieces: "toljda-leklkn" moves (dx, dy); "toljda-rivmdg" moves (-dx, dy); moved in that
    order; a move is skipped if it leaves the grid, reverted if it lands on a wall.
  - Swap rule: horizontally adjacent pieces that cross are both set to the midpoint.
  - ACTION5 and ACTION6 (no 'mosdlc' sprite on level 1) leave the board unchanged.
  - Step bar: rows 0 and 63 show L = round(64*(150-c)/150) cells, c = actions since reset.
    c is NOT in the grid. TICK_POLICY chooses the next bar length from L alone.
No file reads, no simulator, no recorded grids.
"""
import numpy as np

SCALE, OFF, N = 5, 4, 11
FLOOR, PIECE = 5, 10
MAX_STEPS = 150
TICK_POLICY = "no_tick_unless_forced"  # chosen on the 17 visible rows only


def _bar_len_of_count(c):
    return min(int(round(64 * (MAX_STEPS - c) / MAX_STEPS)), 64)


_COUNTS_FOR_LEN = {}
for _c in range(0, MAX_STEPS + 1):
    _COUNTS_FOR_LEN.setdefault(_bar_len_of_count(_c), []).append(_c)


def _next_bar_len(L, policy=None):
    policy = policy or TICK_POLICY
    cs = _COUNTS_FOR_LEN.get(L)
    if not cs:
        return L
    nxt = [_bar_len_of_count(c + 1) for c in cs]
    if policy == "no_tick_unless_forced":
        return L if L in nxt else min(nxt)
    if policy == "always_tick":
        return min(nxt)
    raise ValueError(policy)


def _cell_px(gx, gy):
    return OFF + SCALE * gx, OFF + SCALE * gy  # (x, y) pixel of the cell's top-left


def _pieces(g):
    out = []
    for gy in range(N):
        for gx in range(N):
            x, y = _cell_px(gx, gy)
            if g[y, x] == PIECE:
                out.append((gx, gy))
    return out


def _is_wall(g, gx, gy):
    x, y = _cell_px(gx, gy)
    return int(g[y, x]) not in (FLOOR, PIECE)


def _move(g, pos, dx, dy):
    nx, ny = pos[0] + dx, pos[1] + dy
    if nx < 0 or nx >= N or ny < 0 or ny >= N:
        return pos
    if _is_wall(g, nx, ny):
        return pos
    return (nx, ny)


def _paint(g, pos, color):
    x, y = _cell_px(*pos)
    g[y:y + SCALE, x:x + SCALE] = color


def engine(grid, action, data):
    g = np.array(grid, copy=True)
    # step bar (rows 0 and 63), updated on every action
    L = int(np.sum(g[0] == FLOOR))
    L2 = _next_bar_len(L)
    for x in range(64):
        g[0, x] = FLOOR if x < L2 else 0
        g[63, x] = FLOOR if 63 - x < L2 else 0
    action = int(action)
    if action not in (1, 2, 3, 4):
        return g
    dx, dy = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}[action]
    ps = _pieces(np.asarray(grid))
    if len(ps) != 2:
        return g
    # the left piece is "leklkn" (it starts left at (3,9); rivmdg starts at (7,9))
    p1, p2 = sorted(ps)
    n1 = _move(np.asarray(grid), p1, dx, dy)
    n2 = _move(np.asarray(grid), p2, -dx, dy)
    if abs(p1[0] - p2[0]) == 1 and p1[1] == p2[1]:
        if n1 == p2 or n2 == p1:
            m = ((n1[0] + n2[0]) // 2, (n1[1] + n2[1]) // 2)
            n1 = n2 = m
    _paint(g, p1, FLOOR)
    _paint(g, p2, FLOOR)
    _paint(g, n1, PIECE)
    _paint(g, n2, PIECE)
    return g


def is_level_complete(grid):
    ps = _pieces(np.asarray(grid))
    return len(ps) == 1
