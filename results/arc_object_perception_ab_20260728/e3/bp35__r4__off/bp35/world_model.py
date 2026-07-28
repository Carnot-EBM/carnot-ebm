import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 15
        return new_grid
    else:
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r, c] = 10
        return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    return np.all(grid == 5) or np.all(grid == 10)