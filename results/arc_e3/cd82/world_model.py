import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        if data is not None:
            r, c = data['y'], data['x']
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 5
        else:
            new_grid[63, 63] = 5
    elif action == 3:
        if data is not None:
            r, c = data['y'], data['x']
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 5
        else:
            new_grid[63, 63] = 5
    elif action == 5:
        if data is not None:
            r, c = data['y'], data['x']
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 5
        else:
            new_grid[63, 63] = 5
    elif action == 6:
        if data is not None:
            r, c = data['y'], data['x']
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 5
        else:
            new_grid[63, 63] = 5
    
    return new_grid

def is_level_complete(grid):
    return grid[63, 63] == 5