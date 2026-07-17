import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= 64:
            return grid
        if px < 0 or px >= 64:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 12
        return new_grid
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    if grid.shape != (64, 64):
        return False
    if grid.dtype != np.int64:
        return False
    for i in range(64):
        for j in range(64):
            if grid[i, j] != 12:
                return False
    return True