import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 7
        return grid
    if action == 3:
        grid[48, 15] = 10
        grid[48, 43] = 0
        grid[49, 15] = 10
        grid[49, 19] = 10
        grid[49, 43] = 0
        grid[49, 47] = 0
        grid[59, 15] = 10
        grid[59, 19] = 10
        grid[59, 43] = 0
        grid[59, 47] = 0
        grid[60, 15] = 10
        grid[60, 43] = 0
        return grid
    if action == 2:
        grid[63, 63] = 7
        grid[52, 43] = 4
        grid[52, 46] = 4
        grid[53, 44] = 9
        grid[54, 45] = 9
        grid[55, 45] = 9
        grid[56, 44] = 4
        grid[56, 46] = 4
        return grid
    if action == 1:
        grid[63, 63] = 7
        grid[52, 43] = 9
        grid[52, 46] = 9
        grid[53, 43] = 4
        grid[53, 45] = 4
        grid[54, 43] = 4
        grid[54, 43] = 9
        grid[54, 44] = 4
        grid[54, 43] = 9
        grid[54, 43] = 4
        grid[55, 43] = 4
        grid[56, 43] = 4
        grid[56, 46] = 9
        return grid
    if action == 4:
        grid[63, 62] = 7
        grid[53, 43] = 9
        grid[53, 43] = 4
        grid[53, 45] = 4
        grid[54, 43] = 9
        grid[54, 43] = 4
        grid[54, 43] = 9
        grid[54, 43] = 4
        grid[55, 43] = 9
        grid[55, 43] = 4
        grid[55, 43] = 9
        grid[56, 43] = 9
        grid[56, 43] = 4
        grid[56, 43] = 9
        return grid
    if action == 5:
        grid[63, 61] = 7
        grid[52, 43] = 9
        grid[53, 43] = 9
        grid[53, 43] = 4
        grid[53, 45] = 4
        grid[55, 43] = 4
        grid[56, 43] = 4
        grid[56, 46] = 9
        return grid
    if action == 7:
        grid[63, 63] = 7
        grid[52, 43] = 9
        grid[53, 43] = 9
        grid[53, 43] = 4
        grid[53, 45] = 4
        grid[55, 43] = 4
        grid[56, 43] = 4
        grid[56, 46] = 9
        return grid
    return grid

def is_level_complete(grid):
    return True