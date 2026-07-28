import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        if py < H and px < W:
            new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if grid.shape != (64, 64):
        return False
    for r in range(H):
        row = grid[r]
        if row[0] != 1:
            return False
        if row[-1] != 2:
            return False
        for c in range(1, W - 1):
            if grid[r, c] not in [0, 5]:
                return False
    return True