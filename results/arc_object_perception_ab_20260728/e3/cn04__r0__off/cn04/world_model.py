import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 12
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 12:
                return False
    return True