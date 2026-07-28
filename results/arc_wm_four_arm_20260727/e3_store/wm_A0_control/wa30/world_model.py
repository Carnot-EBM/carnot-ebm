import numpy as np

def engine(grid, action, data):
    if action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 64:
            grid[py, px] = 0
            if grid[py - 1, px] == 5:
                grid[py - 1, px] = 7
            if grid[py - 2, px] == 5:
                grid[py - 2, px] = 7
            if grid[py - 3, px] == 5:
                grid[py - 3, px] = 7
            if grid[py - 4, px] == 5:
                grid[py - 4, px] = 7
            if grid[py - 5, px] == 5:
                grid[py - 5, px] = 7
            if grid[py - 6, px] == 5:
                grid[py - 6, px] = 7
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 64:
            grid[py, px] = 0
            if grid[py - 1, px] == 5:
                grid[py - 1, px] = 7
            if grid[py - 2, px] == 5:
                grid[py - 2, px] = 7
            if grid[py - 3, px] == 5:
                grid[py - 3, px] = 7
            if grid[py - 4, px] == 5:
                grid[py - 4, px] = 7
            if grid[py - 5, px] == 5:
                grid[py - 5, px] = 7
            if grid[py - 6, px] == 5:
                grid[py - 6, px] = 7
    return grid

def is_level_complete(grid):
    return True