import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 5:
            grid[py, px] = 15
            grid[py, px-1] = 15
            grid[py, px+1] = 15
            grid[py-1, px] = 15
            grid[py+1, px] = 15
            grid[py-1, px-1] = 15
            grid[py-1, px+1] = 15
            grid[py+1, px-1] = 15
            grid[py+1, px+1] = 15
            return grid
        else:
            return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5 and grid[r, c] != 15:
                return False
    return True