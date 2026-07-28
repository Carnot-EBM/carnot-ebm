import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 1
            return grid
        return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        row = grid[r, :]
        if np.all(row == 4):
            continue
        if np.all(row == 5):
            continue
        if np.all(row == 0):
            continue
        if np.all(row == 2):
            continue
        if np.all(row == 14):
            continue
        if np.all(row == 15):
            continue
        if np.all(row == 11):
            continue
        if np.all(row == 9):
            continue
        if np.all(row == 8):
            continue
        if np.all(row == 3):
            continue
        if np.all(row == 1):
            continue
        return False
    return True