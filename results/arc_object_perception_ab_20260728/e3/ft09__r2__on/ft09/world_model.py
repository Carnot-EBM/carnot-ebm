import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 5:
            return grid
        grid = grid.copy()
        grid[py, px] = 8
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid[0:64, 0:60] == 4) and np.all(grid[0:64, 60:64] == 9) and np.all(grid[63, :] == 12)