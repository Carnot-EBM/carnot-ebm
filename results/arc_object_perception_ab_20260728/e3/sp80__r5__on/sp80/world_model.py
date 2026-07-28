import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 0 or logical_y >= H or logical_x < 0 or logical_x >= W:
            return grid
        grid_copy = grid.copy()
        grid_copy[logical_y, logical_x] = 0
        return grid_copy
    elif action == 5:
        grid_copy = grid.copy()
        for r in range(H):
            for c in range(W):
                grid_copy[r, c] = 1
        return grid_copy
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 1:
                return False
    return True