import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    if action == 6 and data is not None:
        try:
            px = int(data.get('x', 0))
            py = int(data.get('y', 0))
        except Exception:
            px, py = 0, 0
        if 0 <= py < g.shape[0] and 0 <= px < g.shape[1]:
            if g[py, px] == 6:
                g[0, 0] = 5
    return g

def is_level_complete(grid):
    return False