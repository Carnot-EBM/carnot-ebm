import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        row = grid[r, :]
        if row[0] != 1:
            return False
        for c in range(1, w):
            if row[c] != 2 and row[c] != 5:
                return False
    return True