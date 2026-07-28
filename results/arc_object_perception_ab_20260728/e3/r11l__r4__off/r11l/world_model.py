import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if r == py and c == px:
                    new_grid[r, c] = 1
        return new_grid
    return grid

def is_level_complete(grid):
    return False