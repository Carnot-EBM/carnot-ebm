import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 6
        return new_grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 6
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(64):
        for c in range(64):
            if grid[r, c] != 6:
                return False
    return True