import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        col = py // 1
        row = px // 1
        if row >= H or col >= W:
            return grid
        new_grid = grid.copy()
        for r in range(row, H):
            if r == row:
                new_grid[r, col] = 5
            else:
                new_grid[r, col] = 9
        return new_grid
    elif action == 5:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        col = py // 1
        row = px // 1
        if row >= H or col >= W:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 0
        return new_grid
    elif action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        col = py // 1
        row = px // 1
        if row >= H or col >= W:
            return grid
        new_grid = grid.copy()
        for r in range(row, H):
            if r == row:
                new_grid[r, col] = 11
            else:
                new_grid[r, col] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        row_vals = grid[r, :]
        if len(np.unique(row_vals)) > 3:
            return False
    for r in range(H):
        row_vals = grid[r, :]
        if np.sum(row_vals == 5) < 10:
            return False
    for r in range(H):
        row_vals = grid[r, :]
        if np.sum(row_vals == 5) > 50:
            return False
    return True