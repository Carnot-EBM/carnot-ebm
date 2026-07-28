import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        r, c = py, px
        if r < 0 or r >= H or c < 0 or c >= W:
            return grid
        if grid[r, c] == 5:
            return grid
        grid[r, c] = 5
        return grid
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        r, c = py, px
        if r < 0 or r >= H or c < 0 or c >= W:
            return grid
        if grid[r, c] == 5:
            return grid
        grid[r, c] = 5
        return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5 and grid[r, c] != 6 and grid[r, c] != 10 and grid[r, c] != 15:
                return False
    return True

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.shape != (10, 10):
        return False
    return np.all(g == 0)
