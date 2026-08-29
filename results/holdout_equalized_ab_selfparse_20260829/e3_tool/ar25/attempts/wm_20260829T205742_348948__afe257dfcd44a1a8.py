import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 2:
        # move the 5-colored block down by 1 (shift rows)
        g[1:, :] = g[:-1, :]
        g[0, :] = 9
    elif action == 3:
        # move the 5-colored block up by 1
        g[:-1, :] = g[1:, :]
        g[-1, :] = 9
    elif action == 4:
        # move left
        g[:, 1:] = g[:, :-1]
        g[:, 0] = 9
    elif action == 5:
        # move right
        g[:, :-1] = g[:, 1:]
        g[:, -1] = 9
    elif action == 6:
        # click: toggle clicked cell between 0 and 5
        if data:
            x = int(data.get('x', 0)); y = int(data.get('y', 0))
            if 0 <= y < H and 0 <= x < W:
                g[y, x] = 0 if g[y, x] == 5 else 5
    return g

def is_level_complete(grid):
    return False
