import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_y < 64 and logical_x < 64:
            grid[logical_y, logical_x] = 14
            grid[logical_y, logical_x + 1] = 14
            grid[logical_y, logical_x + 2] = 14
            grid[logical_y, logical_x + 3] = 14
            grid[63, logical_x] = 4
            grid[63, logical_x + 1] = 4
            grid[63, logical_x + 2] = 4
            grid[63, logical_x + 3] = 4
    return grid

def is_level_complete(grid):
    if grid[0, 0] != 5:
        return False
    if grid[63, 0] != 3:
        return False
    if grid[63, 63] != 4:
        return False
    for r in range(64):
        if grid[r, 0] != 5:
            return False
    for c in range(64):
        if grid[63, c] != 3:
            return False
    return True

import numpy as np

def is_level_complete(grid):
    grid = np.array(grid)
    if grid.shape[0] != 5 or grid.shape[1] != 5:
        return False
    return np.all(grid == 0)
