import numpy as np

def _find_ring(grid):
    # find the 5x5 ring of 9s (center cell != 9)
    H, W = grid.shape
    for r in range(H - 4):
        for c in range(W - 4):
            block = grid[r:r+5, c:c+5]
            if block[2, 2] != 9:
                ring = np.ones((5, 5), dtype=bool)
                ring[2, 2] = False
                if np.all(block[ring] == 9):
                    return (r, c)
    return None

def engine(grid, action, data):
    g = grid.copy()
    pos = _find_ring(g)
    if pos is None:
        return g
    r, c = pos
    dr, dc = 0, 0
    if action == 2:
        dr = 6
    elif action == 4:
        dc = 6
    elif action == 1:
        dr = -6
    elif action == 3:
        dc = -6
    else:
        return g
    nr, nc = r + dr, c + dc
    H, W = g.shape
    if nr < 0 or nc < 0 or nr + 5 > H or nc + 5 > W:
        return g
    # clear old ring cells (set to 5)
    for i in range(5):
        for j in range(5):
            if not (i == 2 and j == 2):
                g[r + i, c + j] = 5
    # place new ring (9)
    for i in range(5):
        for j in range(5):
            if not (i == 2 and j == 2):
                g[nr + i, nc + j] = 9
    return g

def is_level_complete(grid):
    return False
