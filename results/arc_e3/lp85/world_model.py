import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        row_idx = py
        col_idx = px
        if grid[row_idx, col_idx] == 14:
            return grid
        if grid[row_idx, col_idx] != 3:
            return grid
        new_grid = grid.copy()
        new_grid[row_idx, col_idx] = 14
        for r in range(H):
            if r == row_idx:
                continue
            if grid[r, col_idx] == 3:
                new_grid[r, col_idx] = 14
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 3:
                return False
    return True