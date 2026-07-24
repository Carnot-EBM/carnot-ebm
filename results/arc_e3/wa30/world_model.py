import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 3:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    new_grid[r, c] = 7
                    new_grid[r, c] = 0
                    break
    elif action == 2:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 0:
                    new_grid[r, c] = 7
                    new_grid[r, c] = 0
                    break
    return new_grid

def is_level_complete(grid):
    return True