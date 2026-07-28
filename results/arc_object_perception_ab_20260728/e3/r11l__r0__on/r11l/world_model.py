import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        if 0 <= py < h and 0 <= px < w:
            new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 5 and grid[r, c] != 0 and grid[r, c] != 2:
                return False
    return True