import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                grid[py, px] = 0
    elif action == 3:
        grid[38, 17:23] = 5
        grid[39, 17:23] = 5
    elif action == 4:
        grid[38, 17] = 8
        grid[38, 18] = 7
        grid[38, 19] = 8
        grid[38, 20] = 7
        grid[39, 17] = 7
        grid[39, 18] = 8
        grid[39, 19] = 7
        grid[39, 20] = 8
    elif action == 1:
        grid[30, 11:17] = 3
        grid[31, 11] = 3
        grid[31, 12:16] = 0
        grid[31, 16] = 3
        grid[32, 11] = 3
        grid[32, 12] = 0
        grid[32, 13] = 3
        grid[32, 14] = 0
        grid[32, 15] = 3
        grid[32, 16] = 0
        grid[32, 17] = 7
        grid[32, 18] = 8
        grid[32, 19] = 7
        grid[32, 20] = 8
        grid[32, 21] = 7
        grid[32, 22] = 8
        grid[33, 11] = 3
        grid[33, 12] = 0
        grid[33, 13] = 3
        grid[33, 14] = 0
        grid[33, 15] = 3
        grid[33, 16] = 0
        grid[33, 17] = 8
        grid[33, 18] = 7
        grid[33, 19] = 8
        grid[33, 20] = 7
        grid[33, 21] = 8
        grid[33, 22] = 8
        grid[34, 11] = 3
        grid[34, 12] = 0
        grid[34, 13] = 3
        grid[34, 14] = 0
        grid[34, 15] = 3
        grid[34, 16] = 0
        grid[34, 17] = 8
        grid[34, 18] = 7
        grid[34, 19] = 8
        grid[34, 20] = 7
        grid[34, 21] = 8
        grid[34, 22] = 8
        grid[35, 11:17] = 3
        grid[36, 11:13] = 10
        grid[36, 13] = 2
        grid[36, 14:16] = 10
        grid[37, 11:13] = 10
        grid[37, 13] = 2
        grid[37, 14:16] = 10
        grid[38, 11:13] = 10
        grid[38, 13] = 8
        grid[38, 14:16] = 10
        grid[38, 17:23] = 5
        grid[39, 11:13] = 10
        grid[39, 13] = 8
        grid[39, 14:16] = 10
        grid[39, 17:23] = 5
        grid[40, 11:17] = 10
        grid[41, 11:17] = 10
    elif action == 7:
        grid[32, 17:23] = 5
        grid[33, 17:23] = 5
    return grid

def is_level_complete(grid):
    return False