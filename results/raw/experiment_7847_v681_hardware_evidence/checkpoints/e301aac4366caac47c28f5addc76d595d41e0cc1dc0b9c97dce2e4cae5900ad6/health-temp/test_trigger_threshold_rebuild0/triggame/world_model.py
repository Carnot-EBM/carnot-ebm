import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    rs, cs = np.where(g == 2)
    if action == 1 and len(rs):
        g[rs[0], cs[0]] = 0
        g[rs[0], min(g.shape[1] - 1, cs[0] + 1)] = 2
    return g

def is_level_complete(grid):
    return False
