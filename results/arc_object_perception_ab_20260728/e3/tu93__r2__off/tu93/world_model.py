import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 5:
            grid[py, px] = 6
            return grid
        return grid
    if action == 2:
        grid[27, 33] = 9
        grid[28, 33] = 9
        grid[29, 33] = 4
        grid[33, 33] = 9
        grid[34, 33] = 9
        grid[35, 33] = 4
        grid[63, 55] = 0
        return grid
    if action == 3:
        grid[33, 27] = 9
        grid[34, 27] = 4
        grid[34, 27] = 9
        grid[35, 27] = 9
        grid[33, 33] = 0
        grid[34, 33] = 0
        grid[35, 33] = 0
        grid[63, 52] = 0
        return grid
    if action == 4:
        grid[39, 21] = 9
        grid[40, 21] = 0
        grid[41, 21] = 0
        grid[41, 27] = 9
        grid[63, 49] = 0
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 5)