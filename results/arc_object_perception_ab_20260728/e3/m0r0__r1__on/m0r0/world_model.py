import numpy as np

def engine(grid, action, data):
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        r, c = py, px
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = 5
        return grid
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        r, c = py, px
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = 10
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 5)