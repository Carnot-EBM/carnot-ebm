import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 11
        return grid
    
    if action == 1:
        grid = grid.copy()
        h, w = grid.shape
        for row in range(h):
            for col in range(w):
                if grid[row, col] == 4:
                    grid[row, col] = 5
        return grid
    
    if action == 3:
        grid = grid.copy()
        h, w = grid.shape
        for row in range(h):
            for col in range(w):
                if grid[row, col] == 5:
                    grid[row, col] = 4
        return grid
    
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    for row in range(h):
        for col in range(w):
            if grid[row, col] == 4:
                return False
    return True