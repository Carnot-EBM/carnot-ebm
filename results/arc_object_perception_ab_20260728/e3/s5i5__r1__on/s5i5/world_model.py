import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        row = py
        col = px
        if grid[row, col] == 5:
            grid[row, col] = 14
            if grid[row-1, col] == 5:
                grid[row-1, col] = 14
            if grid[row+1, col] == 5:
                grid[row+1, col] = 14
            if grid[row, col-1] == 5:
                grid[row, col-1] = 14
            if grid[row, col+1] == 5:
                grid[row, col+1] = 14
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5 and r < 45:
                return False
    return True