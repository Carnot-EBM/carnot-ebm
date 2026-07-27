import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= grid.shape[1] or py < 0 or py >= grid.shape[0]:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 7:
            return grid
        if grid[py, px] == 13:
            return grid
        if grid[py, px] == 8:
            return grid
        if grid[py, px] == 10:
            return grid
        if grid[py, px] == 5:
            return grid
        if grid[py, px] == 0:
            return grid
        if grid[py, px] == 1:
            return grid
        return grid
    return grid

def is_level_complete(grid):
    return False