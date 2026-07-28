import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            grid[py, px] = 10
    return grid

def is_level_complete(grid):
    return False