import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                grid[py, px] = 10
    elif action in [1, 3, 4, 7]:
        if action == 1:
            rows = [30, 31, 32, 33, 34, 35]
        elif action == 3:
            rows = [38, 39]
        elif action == 4:
            rows = [38, 39]
        elif action == 7:
            rows = [32, 33]
        for r in rows:
            if r < H:
                grid[r, :] = 10
    return grid

def is_level_complete(grid):
    return False