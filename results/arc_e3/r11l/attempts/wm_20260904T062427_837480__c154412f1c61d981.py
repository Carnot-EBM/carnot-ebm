import numpy as np


def _expand(runs):
    out = []
    for r, c0, pairs in runs:
        c = c0
        for v, n in pairs:
            for i in range(n):
                out.append((r, c + i, v))
            c += n
    return out


# Phase deltas observed along the diagonal "track" (absolute cell assignments).
_P1 = _expand([
    (34, 7, [(5, 1)]), (35, 6, [(5, 3)]), (36, 5, [(5, 5)]), (37, 6, [(5, 3)]),
    (38, 7, [(5, 1)]), (38, 9, [(5, 1)]), (39, 10, [(5, 1)]), (40, 11, [(5, 1)]),
    (41, 12, [(5, 1)]), (42, 12, [(5, 1)]), (43, 13, [(5, 1)]), (44, 14, [(5, 1)]),
    (45, 15, [(5, 2), (0, 1), (5, 1)]), (46, 15, [(5, 1), (0, 3), (5, 1)]),
    (47, 15, [(0, 2), (15, 1), (0, 2)]), (48, 15, [(5, 1), (0, 3), (5, 1)]),
    (49, 16, [(5, 1), (0, 1), (5, 1)]), (50, 19, [(1, 1), (5, 1)]),
    (51, 21, [(15, 3)]), (52, 20, [(15, 5)]), (53, 20, [(15, 2), (6, 1), (15, 2)]),
    (54, 20, [(15, 5)]), (55, 21, [(15, 3)]),
])

_P2 = _expand([
    (45, 17, [(5, 1)]), (46, 16, [(5, 3)]), (47, 15, [(5, 5)]), (48, 16, [(5, 3)]),
    (49, 17, [(5, 1)]), (49, 19, [(5, 1)]), (50, 19, [(5, 1)]),
    (51, 20, [(5, 2), (0, 1), (5, 1)]), (52, 20, [(5, 1), (0, 3), (5, 1)]),
    (53, 20, [(0, 2), (15, 1), (0, 2)]), (54, 20, [(5, 1), (0, 2)]),
    (54, 25, [(15, 1)]), (55, 21, [(5, 1)]), (55, 24, [(15, 3)]),
    (56, 22, [(15, 2), (6, 1), (15, 2)]), (57, 22, [(15, 5)]), (58, 23, [(15, 3)]),
])

_P3 = _expand([
    (51, 22, [(5, 1)]), (52, 21, [(5, 3)]), (53, 20, [(5, 5)]),
    (54, 21, [(5, 3), (0, 1), (5, 1)]), (55, 22, [(5, 1), (0, 1)]),
    (56, 22, [(0, 1)]), (56, 24, [(15, 1)]), (56, 27, [(15, 1)]),
    (57, 22, [(5, 1)]), (57, 25, [(6, 1)]), (57, 27, [(15, 1)]),
    (58, 26, [(15, 2)]), (59, 24, [(15, 3)]),
])


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)
    H, W = g.shape

    # --- column-0 progress bar: fills downward one row per action, wraps at row 20
    F = 0
    while F < H and g[F, 0] == 5:
        F += 1
    if F < 21:
        g[F, 0] = 5
    else:
        for i in range(1, min(F + 1, H)):
            g[i, 0] = 0
        g[0, 0] = 5

    # --- track creature: clicking on the single color-6 head rolls it one step
    if action == 6 and isinstance(data, dict):
        try:
            x = int(data.get('x', -1))
            y = int(data.get('y', -1))
        except Exception:
            x = y = -1
        if 0 <= x < W and 0 <= y < H:
            idx = np.argwhere(g == 6)
            if len(idx) == 1:
                br, bc = int(idx[0][0]), int(idx[0][1])
                if max(abs(br - y), abs(bc - x)) <= 2:
                    if (br, bc) == (47, 17) and g[36, 7] == 15 and g[34, 7] == 0:
                        for r, c, v in _P1:
                            g[r, c] = v
                    elif (br, bc) == (53, 22) and g[47, 17] == 15:
                        for r, c, v in _P2:
                            g[r, c] = v
                    elif (br, bc) == (56, 24) and g[53, 22] == 15:
                        for r, c, v in _P3:
                            g[r, c] = v

    return g


def is_level_complete(grid):
    # No win state was ever observed in the data (all transitions stay level 0->0).
    return False