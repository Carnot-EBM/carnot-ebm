import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action == 6 and data is not None:
        x = int(data.get('x', 0)); y = int(data.get('y', 0))
        # find the 6 cell
        six = np.argwhere(g == 6)
        if len(six) == 1:
            r, c = int(six[0][0]), int(six[0][1])
            # move 6 down-right by 1 (along the 1-diagonal)
            nr, nc = r + 1, c + 1
            if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                g[r, c] = 15
                g[nr, nc] = 6
    return g

def is_level_complete(grid):
    return False
