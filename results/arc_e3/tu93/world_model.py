import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
    elif action == 4:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
    elif action == 2:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 0
    elif action == 1:
        pass
    elif action == 5:
        pass
    elif action == 7:
        pass
        
    return new_grid

def is_level_complete(grid):
    return False