import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if r == py and c == px:
                    new_grid[r, c] = 5
        return new_grid
    elif action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if r == py and c == px:
                    new_grid[r, c] = 8
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        row = grid[r]
        if np.any(row == 0) or np.any(row == 11) or np.any(row == 12):
            return False
        if np.any(row != 5) and np.any(row != 6) and np.any(row != 15):
            return False
    return True