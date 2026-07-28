import numpy as np

def engine(grid, action, data):
    if action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 0
        return grid
    elif action == 4:
        grid = grid.copy()
        grid[15, 15:18] = 0
        grid[15, 21:24] = 4
        grid[16, 15:18] = 0
        grid[16, 21:24] = [4, 3]
        grid[17, 15:18] = 0
        grid[17, 21:24] = 4
        grid[63, 61:63] = 0
        return grid
    elif action == 2:
        grid = grid.copy()
        grid[15, 21:24] = 0
        grid[16, 21:24] = 0
        grid[17, 21:24] = 0
        grid[21, 21:24] = 4
        grid[22, 21:24] = 4
        grid[23, 21:24] = [4, 3, 4]
        grid[63, 60:61] = 0
        return grid
    elif action == 2:
        grid = grid.copy()
        grid[21, 21:24] = 0
        grid[22, 21:24] = 0
        grid[23, 21:24] = 0
        grid[27, 21:24] = 4
        grid[28, 21:24] = 4
        grid[29, 21:24] = [4, 3, 4]
        grid[63, 59:60] = 0
        return grid
    elif action == 2:
        grid = grid.copy()
        grid[63, 58:59] = 0
        return grid
    elif action == 3:
        grid = grid.copy()
        grid[63, 56:58] = 0
        return grid
    else:
        return grid

def is_level_complete(grid):
    return False