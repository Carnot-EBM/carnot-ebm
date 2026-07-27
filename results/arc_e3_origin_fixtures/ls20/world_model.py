import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        for r in range(45, 50):
            grid[r, 29:34] = 5
            grid[r, 34:39] = 11
        grid[61, 13:24] = 11
        grid[62, 13:24] = 11
    elif action == 2:
        for c in range(14, 19):
            grid[61, c:c+1] = 11
            grid[62, c:c+1] = 11
    return grid

def is_level_complete(grid):
    return False