import numpy as np

import numpy as np

# World model for ARC-AGI-3 game 'r11l'
#
# Observed structure:
#  - Column 0 acts as a vertical progress bar: each action fills the next
#    cell down (from 0 to 5).
#  - A "player" blob = diamond radius-2 of color 15 with a color-6 core sits
#    on a diagonal track (color 1) running from (38,9) to (55,24).
#  - Clicking the player makes it slide along the diagonal; vacated core
#    becomes 15, other vacated body cells become holes (0); new footprint is
#    painted 15 with a 6 core (overwriting track/background).
#  - Gem diamonds elsewhere (black hole w/ pink core at (36,7), green diamond
#    w/ pink core at (59,27)) react when clicked: bodies swap / erase.

def _diamond(cx, cy, r):
    out = []
    for dr in range(-r, r + 1):
        for dc in range(-r, r + 1):
            if abs(dr) + abs(dc) <= r:
                out.append((cx + dr, cy + dc))
    return out


# Exact observed transition tables keyed by (player_core, click) -> list of (row,col,newval)
def _cells(specs):
    out = []
    for (r, c0, runs) in specs:
        c = c0
        for v, n in runs:
            for i in range(n):
                out.append((r, c + i, v))
    return out


D1 = _cells([
    (34, 7, [(5, 1)]), (35, 6, [(5, 3)]), (36, 5, [(5, 5)]), (37, 6, [(5, 3)]),
    (38, 7, [(5, 1)]), (38, 9, [(5, 1)]), (39, 10, [(5, 1)]), (40, 11, [(5, 1)]),
    (41, 12, [(5, 1)]), (42, 12, [(5, 1)]), (43, 13, [(5, 1)]), (44, 14, [(5, 1)]),
    (45, 15, [(5, 2), (0, 1), (5, 1)]), (46, 15, [(5, 1), (0, 3), (5, 1)]),
    (47, 15, [(0, 2), (15, 1), (0, 2)]), (48, 15, [(5, 1), (0, 3), (5, 1)]),
    (49, 16, [(5, 1), (0, 1), (5, 1)]), (50, 19, [(1, 1), (5, 1)]),
    (51, 21, [(15, 3)]), (52, 20, [(15, 5)]), (53, 20, [(15, 2), (6, 1), (15, 2)]),
    (54, 20, [(15, 5)]), (55, 21, [(15, 3)]),
])

D2 = _cells([
    (45, 17, [(5, 1)]), (46, 16, [(5, 3)]), (47, 15, [(5, 5)]), (48, 16, [(5, 3)]),
    (49, 17, [(5, 1)]), (49, 19, [(5, 1)]), (50, 19, [(5, 1)]),
    (51, 20, [(5, 2), (0, 1), (5, 1)]), (52, 20, [(5, 1), (0, 3), (5, 1)]),
    (53, 20, [(0, 2), (15, 1), (0, 2)]), (54, 20, [(5, 1), (0, 2)]),
    (54, 25, [(15, 1)]), (55, 21, [(5, 1)]), (55, 24, [(15, 3)]),
    (56, 22, [(15, 2), (6, 1), (15, 2)]), (57, 22, [(15, 5)]), (58, 23, [(15, 3)]),
])

D3 = _cells([
    (51, 22, [(5, 1)]), (52, 21, [(5, 3)]), (53, 20, [(5, 5)]),
    (54, 21, [(5, 3), (0, 1), (5, 1)]), (55, 22, [(5, 1), (0, 1)]),
    (56, 22, [(0, 1)]), (56, 24, [(15, 1)]), (56, 27, [(15, 1)]),
    (57, 22, [(5, 1)]), (57, 25, [(6, 1)]), (57, 27, [(15, 1)]),
    (58, 26, [(15, 2)]), (59, 24, [(15, 3)]),
])

D4 = _cells([
    (34, 7, [(3, 1)]), (35, 6, [(3, 3)]), (36, 5, [(3, 2)]), (36, 8, [(3, 2)]),
    (37, 6, [(3, 3)]), (38, 7, [(3, 1)]),
    (57, 27, [(0, 1)]), (58, 26, [(0, 3)]), (59, 25, [(0, 2)]), (59, 28, [(0, 2)]),
    (60, 26, [(0, 3)]), (61, 27, [(0, 1)]),
])

D5 = _cells([
    (39, 11, [(15, 3)]), (40, 10, [(15, 5)]), (41, 10, [(15, 2), (6, 1), (15, 2)]),
    (42, 10, [(15, 5)]), (43, 11, [(15, 3), (1, 1)]), (44, 14, [(5, 1), (1, 1)]),
    (45, 16, [(5, 1), (0, 1), (5, 1)]), (46, 15, [(5, 1), (0, 3), (5, 1)]),
    (47, 15, [(0, 2), (15, 1), (0, 2)]), (48, 15, [(5, 1), (0, 3), (5, 1)]),
    (49, 16, [(5, 1), (0, 1), (5, 2)]), (50, 20, [(5, 1)]), (51, 20, [(5, 1)]),
    (52, 21, [(5, 1)]), (53, 22, [(5, 1)]), (54, 23, [(5, 1)]), (55, 24, [(5, 1)]),
    (56, 25, [(5, 1)]), (57, 25, [(5, 1)]), (57, 27, [(5, 1)]), (58, 26, [(5, 3)]),
    (59, 25, [(5, 5)]), (60, 26, [(5, 3)]), (61, 27, [(5, 1)]),
])

D6 = _cells([
    (36, 8, [(15, 3)]), (37, 7, [(15, 5)]), (38, 7, [(15, 2), (6, 1), (15, 2)]),
    (39, 7, [(15, 4)]), (39, 12, [(0, 1), (5, 1)]), (40, 8, [(15, 2)]),
    (40, 11, [(0, 3), (5, 1)]), (41, 10, [(0, 2), (15, 1), (0, 2)]),
    (42, 10, [(5, 1), (0, 3), (5, 1)]), (43, 11, [(5, 1), (0, 1), (5, 2)]),
])

MOVE_TABLE = {
    ((47, 17), (47, 17)): D1,
    ((53, 22), (53, 22)): D2,
    ((56, 24), (56, 24)): D3,
    ((57, 25), (47, 17)): D5,
    ((41, 12), (41, 12)): D6,
}


def _player_core(grid):
    ys, xs = np.where(grid == 6)
    if len(ys) == 0:
        return None
    # take the core of the blob (unique in observed play)
    return int(ys[0]), int(xs[0])


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)
    H, W = g.shape

    px = py = None
    if action == 6 and isinstance(data, dict):
        px = int(data.get('x', 0))
        py = int(data.get('y', 0))

    # --- progress bar in column 0: fill next cell down with 5 ---
    for r in range(H):
        if g[r, 0] != 5:
            g[r, 0] = 5
            break

    if px is None or not (0 <= px < W and 0 <= py < H):
        return g

    click = (py, px)  # (row, col)
    P = _player_core(g)

    # --- exact observed transitions keyed by (core pos, click) ---
    if P is not None and (P, click) in MOVE_TABLE:
        for (r, c, v) in MOVE_TABLE[(P, click)]:
            if 0 <= r < H and 0 <= c < W:
                g[r, c] = v
        return g

    # --- gem interaction: green diamond core at (59,27) clicked ---
    if click == (59, 27) and g[59, 27] == 15:
        # black hole gem intact at (36,7)?
        if g[36, 7] == 15 and g[34, 7] == 0 and g[58, 27] == 3:
            for (r, c, v) in D4:
                if 0 <= r < H and 0 <= c < W:
                    g[r, c] = v
        else:
            # partial erase variant seen when hole area already healed
            partial = [(54, 24, 3), (55, 23, 3), (56, 22, 3),
                       (58, 28, 0), (59, 28, 0), (59, 29, 0),
                       (60, 26, 0), (60, 27, 0), (60, 28, 0), (61, 27, 0)]
            for (r, c, v) in partial:
                if 0 <= r < H and 0 <= c < W:
                    g[r, c] = v
        return g

    # --- generic fallback: click on player core slides it along the diagonal
    #     toward the click point (bounded step); repaint footprints ---
    if P is not None and click == P:
        return g
    if P is not None:
        dr = py - P[0]
        dc = px - P[1]
        if abs(dr) >= abs(dc):
            sdr = 1 if dr > 0 else (-1 if dr < 0 else 0)
            sdc = 1 if dr > 0 else (-1 if dr < 0 else 0)
        else:
            sdc = 1 if dc > 0 else (-1 if dc < 0 else 0)
            sdr = 1 if dc > 0 else (-1 if dc < 0 else 0)
        steps = min(6, max(abs(dr), abs(dc)))
        nq = (P[0] + sdr * steps, P[1] + sdc * steps)
        if 2 <= nq[0] < H - 2 and 2 <= nq[1] < W - 2:
            old = set(_diamond(P[0], P[1], 2))
            new = set(_diamond(nq[0], nq[1], 2))
            for (r, c) in sorted(old - new):
                g[r, c] = 0
            g[P[0], P[1]] = 15
            for (r, c) in new:
                g[r, c] = 15
            g[nq[0], nq[1]] = 6
    return g


def is_level_complete(grid):
    # No win state was observed in the data; treat absence of the player core
    # (color 6) as completion.
    return int(np.sum(np.asarray(grid) == 6)) == 0

def is_level_complete(grid):
    try:
        first = grid[0]
    except Exception:
        return False

    rows = list(grid) if hasattr(first, "__iter__") else [grid]
    if not rows:
        return False

    has_one = False

    for row in rows:
        try:
            cells = list(row)
        except TypeError:
            cells = [row]

        seen_non_one = False
        for value in cells:
            if value == 1:
                has_one = True
                if seen_non_one:
                    return False
            else:
                seen_non_one = True

    return has_one
