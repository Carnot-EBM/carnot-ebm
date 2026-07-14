import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if r == py and c == px:
                    new_grid[r, c] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 14 and grid[r, c] != 3:
                if grid[r, c] != 4 or grid[r, c] != 5:
                    return False
    return True