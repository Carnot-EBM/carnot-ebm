import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        for r in range(45, 50):
            for c in range(29):
                grid[r, c] = 5
        for r in range(61, 63):
            for c in range(13, 14):
                grid[r, c] = 1
    elif action == 2:
        for r in range(61, 63):
            for c in range(14, 19):
                grid[r, c] = 1
    return grid

def is_level_complete(grid):
    return False