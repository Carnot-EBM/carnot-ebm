import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        new_grid[63, 63] = 5
    elif action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 5
    elif action == 5:
        for r in range(34, 38):
            new_grid[r, 27] = 8
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 5
    
    return new_grid

def is_level_complete(grid):
    return False