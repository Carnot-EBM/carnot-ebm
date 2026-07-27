import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 1
        return grid
    if action == 7:
        grid[63, :] = 1
        return grid
    if action == 4:
        grid[37, 25:37] = 1
        grid[38, 25:37] = 1
        grid[39, 25:37] = 1
        grid[40, 25:37] = 1
        grid[41, 26:37] = 1
        grid[63, 1:7] = 1
        return grid
    if action == 3:
        grid[37, 25:37] = 1
        grid[38, 25:37] = 1
        grid[39, 25:37] = 1
        grid[40, 25:37] = 1
        grid[41, 26:37] = 1
        grid[63, 5:11] = 1
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid[63, :] == 1)