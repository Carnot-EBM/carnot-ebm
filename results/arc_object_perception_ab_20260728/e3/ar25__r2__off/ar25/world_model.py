import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 9:
                    grid[r, c] = 5
    elif action == 2:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 9:
                    grid[r, c] = 5
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 9:
                return False
    return True