import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            grid[py, px] = 15
            grid[py, px-1] = 15
            grid[py, px+1] = 15
            grid[py-1, px] = 15
            grid[py+1, px] = 15
            grid[py, px-2] = 15
            grid[py, px+2] = 15
            grid[py-1, px-1] = 15
            grid[py-1, px+1] = 15
            grid[py+1, px-1] = 15
            grid[py+1, px+1] = 15
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        if grid[r, 0] != 5:
            return False
        for c in range(1, W):
            if grid[r, c] != 5:
                return False
    return True