import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if grid[py, px] == 15:
            grid[py, px] = 10
            grid[py-1, px] = 10
            grid[py-2, px] = 10
            grid[py-3, px] = 10
            grid[py-4, px] = 10
            grid[py-5, px] = 10
        return grid
    return grid

def is_level_complete(grid):
    return False