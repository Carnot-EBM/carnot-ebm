import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 9
        return grid
    if action == 2:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    grid[r, c] = 2
        return grid
    if action == 4:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    grid[r, c] = 4
        return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if np.any(grid != 0):
        return False
    return True