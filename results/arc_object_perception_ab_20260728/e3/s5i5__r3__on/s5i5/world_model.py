import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        if grid[py, px] == 5:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 14
        new_grid[py-1, px] = 14
        new_grid[py+1, px] = 14
        new_grid[py, px-1] = 14
        new_grid[py, px+1] = 14
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5 and grid[r, c] != 15:
                return False
    return True

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (5, 5):
        return False
    return np.all(grid == 0)
