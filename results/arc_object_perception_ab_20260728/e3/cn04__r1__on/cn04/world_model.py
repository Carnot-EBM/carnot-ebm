import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        row, col = py, px
        if grid[row, col] == 10:
            grid[row, col] = 12
            return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 12 and grid[r, c] != 0 and grid[r, c] != 4 and grid[r, c] != 8 and grid[r, c] != 10 and grid[r, c] != 14:
                return False
    return True

import numpy as np

def is_level_complete(grid):
    grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    if not np.array_equal(grid, np.zeros((10, 10), dtype=int)):
        return False
    return True
