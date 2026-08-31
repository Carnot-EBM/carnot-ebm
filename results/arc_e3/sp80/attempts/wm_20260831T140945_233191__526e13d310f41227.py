import numpy as np

MOVE = {1: (-4, 0), 2: (4, 0), 3: (0, -4), 4: (0, 4)}

def _find_block(grid, color):
    mask = grid == color
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return None
    r0, r1 = int(rows[0]), int(rows[-1])
    c0, c1 = int(cols[0]), int(cols[-1])
    if not mask[r0:r1 + 1, c0:c1 + 1].all():
        return None
    return r0, c0, r1 - r0 + 1, c1 - c0 + 1

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    h, w = g.shape
    # 1) the 9-block moves by 4 cells in the action direction (1 up, 2 down, 3 left, 4 right)
    if action in MOVE:
        blk = _find_block(g, 9)
        if blk is not None:
            r0, c0, bh, bw = blk
            dr, dc = MOVE[action]
            nr0, nc0 = r0 + dr, c0 + dc
            if 0 <= nr0 and 0 <= nc0 and nr0 + bh <= h and nc0 + bw <= w:
                g[r0:r0 + bh, c0:c0 + bw] = 12
                g[nr0:nr0 + bh, nc0:nc0 + bw] = 9
    # 2) the top bar (row 0, color 14) erodes from the right by 2 cells each step
    if h > 0 and (g[0] == 14).any():
        cols = np.where(g[0] == 14)[0]
        c1 = int(cols[-1])
        g[0, max(0, c1 - 1):c1 + 1] = 0
    return g

def is_level_complete(grid):
    return False
