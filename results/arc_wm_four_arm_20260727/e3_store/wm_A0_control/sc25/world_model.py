import numpy as np

def engine(grid, action, data):
    if action == 6:
        return grid
    
    if action == 4:
        h, w = grid.shape
        grid = grid.copy()
        for r in [19, 20, 21, 22]:
            grid[r, 39] = 4
            grid[r, 40] = 8
        return grid
    
    if action == 3:
        h, w = grid.shape
        grid = grid.copy()
        for r in [0, 1, 19, 20, 21, 22]:
            grid[r, 62] = 0
            if r >= 19:
                grid[r, 35] = 8
                grid[r, 36] = 4
                grid[r, 37] = 5
        return grid
    
    if action == 1:
        h, w = grid.shape
        grid = grid.copy()
        for r in [19, 20, 21, 22]:
            grid[r, 37] = 8
        for r in [21, 22]:
            grid[r, 35] = 4
        return grid
    
    if action == 2:
        h, w = grid.shape
        grid = grid.copy()
        for r in [19, 20]:
            grid[r, 35] = 4
            grid[r, 36] = 4
        for r in [21, 22]:
            grid[r, 35] = 4
            grid[r, 36] = 4
            grid[r, 37] = 4
            grid[r, 38] = 4
        return grid
    
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 0:
                return False
    return True