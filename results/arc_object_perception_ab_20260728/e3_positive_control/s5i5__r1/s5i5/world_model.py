import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 5:
            grid[py, px] = 15
            if py > 0:
                grid[py-1, px] = 5
            if py < H-1:
                grid[py+1, px] = 5
            return grid
        else:
            grid[py, px] = 5
            return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        row = grid[r, :]
        if not np.all(row == 5):
            return False
    return True