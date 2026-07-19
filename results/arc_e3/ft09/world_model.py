import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if px < 0 or px >= w or py < 0 or py >= h:
            return new_grid
            
        if grid[py, px] == 12:
            new_grid[py, px] = 0
            return new_grid
            
        if grid[py, px] == 0:
            new_grid[py, px] = 12
            return new_grid
            
        return new_grid
        
    return new_grid

def is_level_complete(grid):
    return False