import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
        else:
            new_grid[0, 0] = 0
    elif action == 4:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
        else:
            new_grid[0, 0] = 0
    elif action == 2:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
        else:
            new_grid[0, 0] = 0
    else:
        new_grid = grid.copy()
        
    return new_grid

def is_level_complete(grid):
    return False