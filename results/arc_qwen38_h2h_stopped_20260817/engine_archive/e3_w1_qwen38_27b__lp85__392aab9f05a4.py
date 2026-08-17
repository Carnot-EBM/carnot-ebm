import numpy as np


def engine(grid, action, data):
    g = np.array(grid, copy=True)

    if action == 0 and g.ndim == 2 and g.shape[0] >= 64 and g.shape[1] >= 64:
        ty, tx = 19, 12

        # Already-complete states are kept stable.
        target_done = bool(np.all(g[ty:ty + 4, tx:tx + 4] == 11))

        progress_done = False
        col = g[:, 0]
        if col.size > 0 and int(col[0]) == 5:
            mask = col != 5
            run = len(col) if not np.any(mask) else int(np.argmax(mask))
            progress_done = run >= 25

        if not (target_done or progress_done):
            # Closed ring of 4x4 tile positions, clockwise from top-left.
            slots = (
                (19, 12), (19, 18), (19, 24), (19, 30), (19, 36), (19, 42), (19, 48),
                (25, 48), (31, 48), (37, 48),
                (43, 48), (43, 42), (43, 36), (43, 30), (43, 24), (43, 18), (43, 12),
                (37, 12), (31, 12), (25, 12),
            )

            vals = [int(g[y, x]) for y, x in slots]
            new_vals = vals[1:] + vals[:1]

            for (y, x), v in zip(slots, new_vals):
                g[y:y + 4, x:x + 4] = v

            # Advance the left-edge progress marker by one 5-cell block.
            if not np.all(col == 5):
                idx = int(np.argmax(col != 5))
                end = min(idx + 5, g.shape[0])
                g[idx:end, 0] = 5

    return g


def is_level_complete(grid):
    try:
        g = np.asarray(grid)
        if g.ndim != 2 or g.shape[0] < 23 or g.shape[1] < 16:
            return False

        # The movable color-11 tile has reached its marked top-left slot.
        target_done = bool(np.all(g[19:23, 12:16] == 11))

        # Or the left-edge progress marker has advanced far enough.
        progress_done = False
        col = g[:, 0]
        if col.size > 0 and int(col[0]) == 5:
            mask = col != 5
            run = len(col) if not np.any(mask) else int(np.argmax(mask))
            progress_done = run >= 25

        return bool(target_done or progress_done)
    except Exception:
        return False