import numpy as np

def engine(grid, action, data):
    if action == 4:
        if grid[40, 10] == 9 and grid[41, 10] == 9:
            grid[40, 10] = 3
            grid[40, 11] = 2
            grid[41, 10] = 3
            grid[41, 11] = 2
        return grid
    elif action == 3:
        if grid[40, 10] == 9 and grid[41, 10] == 9:
            grid[40, 10] = 2
            grid[40, 11] = 3
            grid[41, 10] = 2
            grid[41, 11] = 3
        return grid
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            if grid[63, px] == 0:
                grid[63, px] = 10
            elif grid[63, px] == 10:
                grid[63, px] = 0
            return grid
    return grid

def is_level_complete(grid):
    return False