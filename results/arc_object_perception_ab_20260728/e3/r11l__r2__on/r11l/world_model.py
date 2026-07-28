import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        if px < 0 or px >= w or py < 0 or py >= h:
            return new_grid
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = py + dy, px + dx
                if 0 <= ny < h and 0 <= nx < w:
                    if new_grid[ny, nx] == 0:
                        new_grid[ny, nx] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for y in range(h):
        for x in range(w):
            if grid[y, x] != 0 and grid[y, x] != 5 and grid[y, x] != 2:
                return False
    for y in range(h):
        row_vals = grid[y, :]
        if np.sum(row_vals == 0) > 1:
            return False
        if np.sum(row_vals == 2) > 1:
            return False
        if np.sum(row_vals == 5) < 1:
            return False
    return True