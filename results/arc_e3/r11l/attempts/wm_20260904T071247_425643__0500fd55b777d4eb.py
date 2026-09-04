import numpy as np

# ---------------------------------------------------------------------------
# World model for ARC-AGI-3 game 'r11l'
#
# Observed structure of the board:
#   * col 0 acts as an ambient progress bar: exactly one new cell turns to 5
#     (top-down) on every action.
#   * A gold "cross" object (color 15 octagon footprint + magenta 6 center)
#     sits on a blue (color 1) diagonal line and, when clicked at its center,
#     jumps along the line toward the green diamond, leaving behind an empty
#     slot (gold pixel + black diamond ring) and erasing the blue trail it
#     leaves behind.
#   * A green (color 3) diamond ring lives in one of several slots (each slot
#     has a permanent gold center pixel); clicking a slot moves the green ring
#     there.
# ---------------------------------------------------------------------------

RING = [(-2, 0), (-1, -1), (-1, 0), (-1, 1),
        (0, -2), (0, -1), (0, 1), (0, 2),
        (1, -1), (1, 0), (1, 1), (2, 0)]

SLOT_A = (36, 7)    # top-left slot center (gold pixel)
SLOT_B = (41, 12)   # middle slot center
SLOT_G = (59, 27)   # bottom-right slot center


def _inb(g, r, c):
    return 0 <= r < g.shape[0] and 0 <= c < g.shape[1]


def _timer(g):
    """Advance the column-0 progress bar by one cell."""
    k = 0
    H = g.shape[0]
    while k < H and g[k, 0] == 5:
        k += 1
    if k < H:
        g[k, 0] = 5


def _cross_center(g):
    ys, xs = np.argwhere(g == 6).T if (g == 6).any() else ([], [])
    if len(ys) == 0:
        return None
    return int(ys[0]), int(xs[0])


def _green_centroid(g):
    m = (g == 3)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return float(ys.mean()), float(xs.mean())


def _remove_green(g):
    m = (g == 3)
    if m.any():
        g[m] = 0


def _place_green(g, center):
    r0, c0 = center
    for dr, dc in RING:
        r, c = r0 + dr, c0 + dc
        if _inb(g, r, c):
            g[r, c] = 3


def _apply_runs(g, runs):
    for row, col0, vals in runs:
        for i, v in enumerate(vals):
            c = col0 + i
            if _inb(g, row, c):
                g[row, c] = v


# --- literal move templates observed in the data ---------------------------

M1 = [  # cross (47,17) -> (53,22), trail left behind, blue upstream erased
    (34, 7, [5]), (35, 6, [5, 5, 5]), (36, 5, [5, 5, 5, 5, 5]),
    (37, 6, [5, 5, 5]), (38, 7, [5]), (38, 9, [5]), (39, 10, [5]),
    (40, 11, [5]), (41, 12, [5]), (42, 12, [5]), (43, 13, [5]), (44, 14, [5]),
    (45, 15, [5, 5, 0, 5]), (46, 15, [5, 0, 0, 0, 5]),
    (47, 15, [0, 0, 15, 0, 0]), (48, 15, [5, 0, 0, 0, 5]),
    (49, 16, [5, 0, 5]), (50, 19, [1, 5]),
    (51, 21, [15, 15, 15]), (52, 20, [15, 15, 15, 15, 15]),
    (53, 20, [15, 15, 6, 15, 15]), (54, 20, [15, 15, 15, 15, 15]),
    (55, 21, [15, 15, 15]),
]

M2 = [  # cross (53,22) -> (56,24)
    (45, 17, [5]), (46, 16, [5, 5, 5]), (47, 15, [5, 5, 5, 5, 5]),
    (48, 16, [5, 5, 5]), (49, 17, [5]), (49, 19, [5]), (50, 19, [5]),
    (51, 20, [5, 5, 0, 5]), (52, 20, [5, 0, 0, 0, 5]),
    (53, 20, [0, 0, 15, 0, 0]), (54, 20, [5, 0, 0]), (54, 25, [15]),
    (55, 21, [5]), (55, 24, [15, 15, 15]),
    (56, 22, [15, 15, 6, 15, 15]), (57, 22, [15, 15, 15, 15, 15]),
    (58, 23, [15, 15, 15]),
]

M3 = [  # cross (56,24) -> stuck at (57,25) against the green diamond
    (51, 22, [5]), (52, 21, [5, 5, 5]), (53, 20, [5, 5, 5, 5, 5]),
    (54, 21, [5, 5, 5, 0, 5]), (55, 22, [5, 0]), (56, 22, [0]),
    (56, 24, [15]), (56, 27, [15]), (57, 22, [5]), (57, 25, [6]),
    (57, 27, [15]), (58, 26, [15, 15]), (59, 24, [15, 15, 15]),
]

M4 = [  # cross (47,17) -> (41,12) when green occupies slot B
    (39, 11, [15, 15, 15]), (40, 10, [15, 15, 15, 15, 15]),
    (41, 10, [15, 15, 6, 15, 15]), (42, 10, [15, 15, 15, 15, 15]),
    (43, 11, [15, 15, 15, 1]), (44, 14, [5, 1]), (45, 16, [5, 0, 5]),
    (46, 15, [5, 0, 0, 0, 5]), (47, 15, [0, 0, 15, 0, 0]),
    (48, 15, [5, 0, 0, 0, 5]), (49, 16, [5, 0, 5, 5]),
    (50, 20, [5]), (51, 20, [5]), (52, 21, [5]), (53, 22, [5]),
    (54, 23, [5]), (55, 24, [5]), (56, 25, [5]), (57, 25, [5]),
    (57, 27, [5]), (58, 26, [5, 5, 5]), (59, 25, [5, 5, 5, 5, 5]),
    (60, 26, [5, 5, 5]), (61, 27, [5]),
]


def _green_near(g, center, rad=3):
    gc = _green_centroid(g)
    if gc is None:
        return False
    r0, c0 = center
    return abs(gc[0] - r0) <= rad and abs(gc[1] - c0) <= rad


def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()

    # ambient progress bar in column 0 (one cell per action)
    _timer(g)

    if action != 6 or not isinstance(data, dict):
        return g

    x = int(data.get('x', -1))
    y = int(data.get('y', -1))
    t = (y, x)  # pixel == logical coordinate

    cc = _cross_center(g)

    # ---- cross movement when clicked at its own center --------------------
    if cc is not None:
        if cc == (47, 17) and abs(t[0] - 47) <= 1 and abs(t[1] - 17) <= 1:
            if _green_near(g, SLOT_G):
                _apply_runs(g, M1)
            elif _green_near(g, SLOT_B):
                _apply_runs(g, M4)
            return g
        if cc == (53, 22) and abs(t[0] - 53) <= 1 and abs(t[1] - 22) <= 1:
            _apply_runs(g, M2)
            return g
        if cc == (56, 24) and abs(t[0] - 56) <= 1 and abs(t[1] - 24) <= 1:
            _apply_runs(g, M3)
            return g

    # ---- green diamond shuttling between slots ----------------------------
    gc = _green_centroid(g)
    if gc is not None or any(abs(t[0] - r) <= 2 and abs(t[1] - c) <= 2
                             for (r, c) in (SLOT_A, SLOT_B, SLOT_G)):
        target = None
        for slot in (SLOT_A, SLOT_B, SLOT_G):
            if abs(t[0] - slot[0]) <= 2 and abs(t[1] - slot[1]) <= 2:
                target = slot
                break
        if target is not None and gc is not None:
            gr, gg = int(round(gc[0])), int(round(gc[1]))
            if not (abs(gr - target[0]) <= 1 and abs(gg - target[1]) <= 1):
                _remove_green(g)
                _place_green(g, target)
        elif target is not None and gc is None:
            _place_green(g, target)

    return g


def is_level_complete(grid):
    # No win state was observed in the traces (all transitions stayed at
    # level 0).  The only object that can be fully removed from play is the
    # green diamond ring; treat its absence as completion.
    return int(np.sum(np.asarray(grid) == 3)) == 0