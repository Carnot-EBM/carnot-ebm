import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        new_grid[py, px] = 0
        for dy in range(1, h):
            if py + dy >= h:
                break
            row = new_grid[py + dy, :]
            if row[px] == 0:
                break
            if row[px] == 5:
                new_grid[py + dy, px] = 0
            else:
                new_grid[py + dy, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0 and grid[r, c] != 5:
                return False
    return True