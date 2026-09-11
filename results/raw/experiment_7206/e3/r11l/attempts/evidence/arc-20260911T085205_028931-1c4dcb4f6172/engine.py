import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        r, c = y, x
        # find the 6 (player)
        pr, pc = None, None
        for i in range(H):
            for j in range(W):
                if g[i, j] == 6:
                    pr, pc = i, j
                    break
            if pr is not None:
                break
        if pr is not None:
            # move player toward click
            dr = 0
            dc = 0
            if r > pr: dr = 1
            elif r < pr: dr = -1
            if c > pc: dc = 1
            elif c < pc: dc = -1
            nr, nc = pr + dr, pc + dc
            if 0 <= nr < H and 0 <= nc < W:
                g[nr, nc] = 6
                g[pr, pc] = 15
    return g

def is_level_complete(grid):
    return False
