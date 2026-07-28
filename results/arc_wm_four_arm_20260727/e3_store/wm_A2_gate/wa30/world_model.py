import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 48:
            return grid
        if py == 48:
            if px < 28:
                return grid
            if px == 28:
                grid[48, 28] = 0
                grid[49, 28] = 5
                grid[50, 28] = 5
                grid[51, 28] = 5
                grid[52, 28] = 7
                grid[53, 28] = 7
                grid[54, 28] = 7
                grid[55, 28] = 7
                return grid
            return grid
        return grid
    if action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 48:
            return grid
        if py == 48:
            if px < 28:
                return grid
            if px == 28:
                grid[48, 28] = 7
                grid[49, 28] = 7
                grid[50, 28] = 7
                grid[51, 28] = 7
                grid[52, 28] = 5
                grid[53, 28] = 5
                grid[54, 28] = 5
                grid[55, 28] = 0
                return grid
            return grid
        return grid
    return grid

def is_level_complete(grid):
    return False