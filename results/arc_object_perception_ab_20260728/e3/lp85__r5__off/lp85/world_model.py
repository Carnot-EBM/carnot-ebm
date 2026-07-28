import numpy as np

def engine(grid, action, data):
    if action == 0:
        if data is None:
            return grid
        h, w = grid.shape
        new_grid = grid.copy()
        if data.get('x') is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 14
            new_grid[py, px + 1] = 14
            new_grid[py + 1, px] = 14
            new_grid[py + 1, px + 1] = 14
            return new_grid
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        row = grid[r, :]
        if row[0] != 14:
            return False
        if row[1] != 3:
            return False
        if row[2] != 10:
            return False
        if row[3] != 4:
            return False
        if row[4] != 41:
            return False
        if row[5] != 3:
            return False
        if row[6] != 12:
            return False
    return True