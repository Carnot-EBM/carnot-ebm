import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 0 or py >= grid.shape[0] or px < 0 or px >= grid.shape[1]:
            return grid
        if grid[py, px] == 15:
            return grid
        if grid[py, px] == 7:
            grid[py, px] = 15
            return grid
        if grid[py, px] == 13:
            grid[py, px] = 15
            return grid
        if grid[py, px] == 0:
            grid[py, px] = 15
            return grid
        if grid[py, px] == 5:
            grid[py, px] = 15
            return grid
        if grid[py, px] == 8:
            grid[py, px] = 15
            return grid
        if grid[py, px] == 10:
            grid[py, px] = 15
            return grid
    return grid

def is_level_complete(grid):
    return grid[63, 0] == 15