import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 0:
            grid[py, px] = 15
            grid[py, px + 1] = 15
            grid[py, px + 2] = 15
            grid[py, px + 3] = 15
            grid[py, px + 4] = 15
            grid[py, px + 5] = 15
            grid[py, px + 6] = 15
            grid[py, px + 7] = 15
            grid[py, px + 8] = 15
            grid[py, px + 9] = 15
            grid[py, px + 10] = 15
            grid[py, px + 11] = 15
            grid[py, px + 12] = 15
            grid[py, px + 13] = 15
            grid[py, px + 14] = 15
            grid[py, px + 15] = 15
            grid[py, px + 16] = 15
            grid[py, px + 17] = 18
            grid[py, px + 18] = 18
            grid[py, px + 19] = 18
            grid[py, px + 20] = 18
            grid[py, px + 21] = 18
            grid[py, px + 22] = 18
            grid[py, px + 23] = 18
            grid[py, px + 24] = 18
            grid[py, px + 25] = 18
            grid[py, px + 26] = 18
            grid[py, px + 27] = 18
            grid[py, px + 28] = 18
            grid[py, px + 29] = 18
            grid[py, px + 30] = 18
            grid[py, px + 31] = 18
            grid[py, px + 32] = 18
            grid[py, px + 33] = 18
            grid[py, px + 34] = 18
            grid[py, px + 35] = 18
            grid[py, px + 36] = 18
            grid[py, px + 37] = 18
            grid[py, px + 38] = 18
            grid[py, px + 39] = 18
            grid[py, px + 40] = 18
            grid[py, px + 41] = 18
            grid[py, px + 42] = 18
            grid[py, px + 43] = 18
            grid[py, px + 44] = 18
            grid[py, px + 45] = 18
            grid[py, px + 46] = 18
            grid[py, px + 47] = 18
            grid[py, px + 48] = 18
            grid[py, px + 49] = 18
            grid[py, px + 50] = 18
            grid[py, px + 51] = 18
            grid[py, px + 52] = 18
            grid[py, px + 53] = 18
            grid[py, px + 54] = 18
            grid[py, px + 55] = 18
            grid[py, px + 56] = 18
            grid[py, px + 57] = 18
            grid[py, px + 58] = 18
            grid[py, px + 59] = 18
            grid[py, px + 60] = 18
            grid[py, px + 61] = 18
            grid[py, px + 62] = 18
            grid[py, px + 63] = 18
            return grid
        else:
            return grid
    return grid

def is_level_complete(grid):
    return False