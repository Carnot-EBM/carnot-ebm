import numpy as np


def engine(grid, action, data):
    H, W = grid.shape
    try:
        dtype = grid.dtype
        if not np.issubdtype(dtype, np.integer):
            dtype = np.int64
    except Exception:
        dtype = np.int64

    out = np.array(grid, dtype=dtype, copy=True)

    def _set_flag(arr, val):
        try:
            arr.__dict__["arc_sp80_complete"] = bool(val)
        except Exception:
            pass
        return arr

    def _next_grid():
        g = np.full((64, 64), 12, dtype=dtype)

        g[0:4, :] = 1

        for r in range(4, 8):
            g[r, 12:24] = 11
            g[r, 28:40] = 11
            g[r, 44:56] = 11

        for r in range(8, 12):
            for c in (12, 20, 28, 36, 44, 52):
                g[r, c:c + 4] = 11

        for r in range(16, 20):
            g[r, 8:20] = 8

        for r in range(24, 28):
            g[r, 28:40] = 8

        for r in range(36, 40):
            g[r, 20:40] = 9

        for r in range(56, 60):
            g[r, 40:44] = 6

        for r in range(60, 63):
            g[r, 40:44] = 4

        g[63, :] = 14
        return g

    def _win_pre(g):
        if g.shape != (64, 64):
            return False

        mask = (g == 9)
        if int(mask.sum()) != 80:
            return False

        rows, cols = np.where(mask)
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())

        if (r0, r1, c0, c1) != (16, 19, 24, 43):
            return False

        if not bool(np.all(g[52:56, 16:20] == 11)):
            return False
        if not bool(np.all(g[52:56, 24:28] == 11)):
            return False
        if not bool(np.all(g[52:56, 40:44] == 11)):
            return False
        if not bool(np.all(g[52:56, 48:52] == 11)):
            return False

        if int(np.sum(g[52:56, 20:24] == 11)) != 0:
            return False
        if int(np.sum(g[52:56, 44:48] == 11)) != 0:
            return False

        if not bool(np.all(g[56:60, 16:28] == 11)):
            return False
        if not bool(np.all(g[56:60, 40:52] == 11)):
            return False

        return True

    try:
        act = int(action)
    except Exception:
        act = -1

    moved = False

    if act in (3, 4):
        dx = 4 if act == 4 else -4
        mask = (out == 9)

        if bool(mask.any()):
            rows, cols = np.where(mask)
            new_cols = cols + dx

            if int(new_cols.min()) >= 0 and int(new_cols.max()) < W:
                dest = np.zeros((H, W), dtype=bool)
                dest[rows, new_cols] = True

                passable = (out == 12) | (out == 0) | (out == 9)

                if not bool(np.any((~passable) & dest)):
                    out[mask] = 12
                    out[dest] = 9
                    moved = True

    if moved:
        row = out[0]
        L = 0
        while L < W and row[L] == 14:
            L += 1

        if L > 0:
            k = 2 if L >= 2 else L
            out[0, L - k:L] = 0

    elif act == 5:
        if _win_pre(out):
            return _set_flag(_next_grid(), True)

    return _set_flag(out, False)


def is_level_complete(grid):
    try:
        d = grid.__dict__
    except Exception:
        d = {}

    if "arc_sp80_complete" in d:
        return bool(d["arc_sp80_complete"])

    shape = getattr(grid, "shape", None)
    if shape != (64, 64):
        return False

    try:
        a = np.asarray(grid).astype(np.int64, copy=False)
    except Exception:
        return False

    g = np.full((64, 64), 12, dtype=np.int64)

    g[0:4, :] = 1

    for r in range(4, 8):
        g[r, 12:24] = 11
        g[r, 28:40] = 11
        g[r, 44:56] = 11

    for r in range(8, 12):
        for c in (12, 20, 28, 36, 44, 52):
            g[r, c:c + 4] = 11

    for r in range(16, 20):
        g[r, 8:20] = 8

    for r in range(24, 28):
        g[r, 28:40] = 8

    for r in range(36, 40):
        g[r, 20:40] = 9

    for r in range(56, 60):
        g[r, 40:44] = 6

    for r in range(60, 63):
        g[r, 40:44] = 4

    g[63, :] = 14

    return bool(np.array_equal(a, g))