import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        if 0 <= py < 64 and 0 <= px < 64:
            grid[py, px] = 8
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 4)