import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 10
        return grid
    elif action == 3:
        grid = grid.copy()
        grid[48, 15:20] = 3
        grid[48, 43:48] = 0
        grid[49, 15:16] = 3
        grid[49, 19:20] = 3
        grid[49, 43:44] = 0
        grid[49, 47:48] = 0
        grid[59, 15:16] = 3
        grid[59, 19:20] = 3
        grid[59, 43:44] = 0
        grid[59, 47:48] = 0
        grid[60, 15:20] = 3
        grid[60, 43:48] = 0
        return grid
    elif action == 2:
        grid = grid.copy()
        if grid[63, 63] != 4:
            grid[63, 63] = 4
        if grid[52, 43] != 7:
            grid[52, 43] = 7
        if grid[52, 46] != 7:
            grid[52, 46] = 7
        if grid[53, 44] != 5:
            grid[53, 44] = 5
        if grid[54, 45] != 5:
            grid[54, 45] = 5
        if grid[55, 45] != 5:
            grid[55, 45] = 5
        if grid[56, 44] != 7:
            grid[56, 44] = 7
        if grid[56, 46] != 7:
            grid[56, 46] = 7
        return grid
    return grid

def is_level_complete(grid):
    return grid[63, 63] == 4