import numpy as np

MOVE = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
STEP = 4
FUEL_COST = 2
FUEL_ROW = 0
FUEL_COLOR = 14
PLAYER_COLOR = 9
BG_COLOR = 12


def _player_bbox(grid):
    mask = grid == PLAYER_COLOR
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return None
    r0 = int(np.argmax(rows))
    r1 = int(len(rows) - 1 - np.argmax(rows[::-1]))
    c0 = int(np.argmax(cols))
    c1 = int(len(cols) - 1 - np.argmax(cols[::-1]))
    return r0, c0, r1, c1


def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    if action in MOVE:
        dr, dc = MOVE[action]
        bb = _player_bbox(g)
        if bb is not None:
            r0, c0, r1, c1 = bb
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            nr0, nc0 = r0 + dr * STEP, c0 + dc * STEP
            nr1, nc1 = r1 + dr * STEP, c1 + dc * STEP
            H, W = g.shape
            if 0 <= nr0 and nr1 < H and 0 <= nc0 and nc1 < W:
                block = g[r0:r1 + 1, c0:c1 + 1].copy()
                # clear old position
                g[r0:r1 + 1, c0:c1 + 1] = BG_COLOR
                # write new position
                g[nr0:nr1 + 1, nc0:nc1 + 1] = block
                # fuel bar shrinks from the right by FUEL_COST
                row = g[FUEL_ROW]
                idx = np.where(row == FUEL_COLOR)[0]
                if idx.size:
                    right = int(idx.max())
                    for k in range(FUEL_COST):
                        c = right - k
                        if c >= 0 and row[c] == FUEL_COLOR:
                            row[c] = 0
    return g


def is_level_complete(grid):
    return False
