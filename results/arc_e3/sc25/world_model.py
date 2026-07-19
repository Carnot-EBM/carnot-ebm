import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        if px < 0 or px >= w or py < 0 or py >= h:
            return grid
        new_grid = grid.copy()
        new_grid[py, px] = 14
        return new_grid
    elif action == 3:
        h, w = grid.shape
        new_grid = grid.copy()
        for row in range(h):
            for col in range(w):
                if new_grid[row, col] == 0:
                    new_grid[row, col] = 14
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    return False