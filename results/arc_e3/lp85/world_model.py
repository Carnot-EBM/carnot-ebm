import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 5
        return grid
    return grid

def is_level_complete(grid):
    return False

import numpy as np

def is_level_complete(grid):
    if grid.shape != (50, 50):
        return False
    return np.all(grid == grid[12:48, 3:47])
