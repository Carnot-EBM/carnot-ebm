import numpy as np

def engine(grid, action, data):
    if action == 3:
        return grid
    if action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = 2
        return new_grid
    return grid

def is_level_complete(grid):
    return False