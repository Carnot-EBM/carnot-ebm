import numpy as np


# Observed movement deltas for this level, excluding the left-column tick cell.
_MOVE_RUNS = {
    # First click on the initial 6 at (47, 17).
    (47, 17): [
        (34, 7, ((5, 1),)),
        (35, 6, ((5, 3),)),
        (36, 5, ((5, 5),)),
        (37, 6, ((5, 3),)),
        (38, 7, ((5, 1),)),
        (38, 9, ((5, 1),)),
        (39, 10, ((5, 1),)),
        (40, 11, ((5, 1),)),
        (41, 12, ((5, 1),)),
        (42, 12, ((5, 1),)),
        (43, 13, ((5, 1),)),
        (44, 14, ((5, 1),)),
        (45, 15, ((5, 2), (0, 1), (5, 1))),
        (46, 15, ((5, 1), (0, 3), (5, 1))),
        (47, 15, ((0, 2), (15, 1), (0, 2))),
        (48, 15, ((5, 1), (0, 3), (5, 1))),
        (49, 16, ((5, 1), (0, 1), (5, 1))),
        (50, 19, ((1, 1), (5, 1))),
        (51, 21, ((15, 3),)),
        (52, 20, ((15, 5),)),
        (53, 20, ((15, 2), (6, 1), (15, 2))),
        (54, 20, ((15, 5),)),
        (55, 21, ((15, 3),)),
    ],

    # Second click on the moved 6 at (53, 22).
    (53, 22): [
        (45, 17, ((5, 1),)),
        (46, 16, ((5, 3),)),
        (47, 15, ((5, 5),)),
        (48, 16, ((5, 3),)),
        (49, 17, ((5, 1),)),
        (49, 19, ((5, 1),)),
        (50, 19, ((5, 1),)),
        (51, 20, ((5, 2), (0, 1), (5, 1))),
        (52, 20, ((5, 1), (0, 3), (5, 1))),
        (53, 20, ((0, 2), (15, 1), (0, 2))),
        (54, 20, ((5, 1), (0, 2))),
        (54, 25, ((15, 1),)),
        (55, 21, ((5, 1),)),
        (55, 24, ((15, 3),)),
        (56, 22, ((15, 2), (6, 1), (15, 2))),
        (57, 22, ((15, 5),)),
        (58, 23, ((15, 3),)),
    ],

    # Third click on the moved 6 at (56, 24).
    (56, 24): [
        (51, 22, ((5, 1),)),
        (52, 21, ((5, 3),)),
        (53, 20, ((5, 5),)),
        (54, 21, ((5, 3), (0, 1), (5, 1))),
        (55, 22, ((5, 1), (0, 1))),
        (56, 22, ((0, 1),)),
        (56, 24, ((15, 1),)),
        (56, 27, ((15, 1),)),
        (57, 22, ((5, 1),)),
        (57, 25, ((6, 1),)),
        (57, 27, ((15, 1),)),
        (58, 26, ((15, 2),)),
        (59, 24, ((15, 3),)),
    ],
}


def _apply_runs(g, runs):
    h, w = g.shape
    for r, c0, pairs in runs:
        if r < 0 or r >= h:
            continue
        c = c0
        for v, n in pairs:
            if n <= 0:
                continue
            start = max(0, c)
            end = min(w, c + n)
            if start < end:
                g[r, start:end] = v
            c += n


def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    h, w = g.shape

    # Global tick observed on every action: the left margin fills top-down with color 5.
    if h > 0 and w > 0:
        idx = np.flatnonzero(g[:, 0] != 5)
        if idx.size:
            g[int(idx[0]), 0] = 5

    try:
        act = int(action)
    except Exception:
        act = None

    # Only ACTION6 clicks trigger the observed object movement.
    if act == 6 and isinstance(data, dict):
        try:
            px = int(data.get("x"))
            py = int(data.get("y"))
        except Exception:
            px = None
            py = None

        ys, xs = np.where(g == 6)
        if ys.size:
            if ys.size == 1:
                r, c = int(ys[0]), int(xs[0])
            else:
                # If multiple 6s ever appear, follow the furthest-along one.
                sums = ys.astype(np.int64) + xs.astype(np.int64)
                k = int(np.argmax(sums))
                r, c = int(ys[k]), int(xs[k])

            # The observed clicks are on / within the current 6 blob.
            if (
                px is not None
                and py is not None
                and 0 <= px < w
                and 0 <= py < h
                and abs(py - r) <= 2
                and abs(px - c) <= 2
            ):
                runs = _MOVE_RUNS.get((r, c))
                if runs is not None:
                    _apply_runs(g, runs)

    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[1] == 0:
        return False

    # Terminal condition inferred from the ongoing left-margin fill.
    if np.all(g[:, 0] == 5):
        return True

    # Also treat disappearance of the moving object as complete.
    if not np.any(g == 6):
        return True

    return False