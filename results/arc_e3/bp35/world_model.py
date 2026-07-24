import numpy as np

def engine(grid, action, data):
    if action == 3:
        h, w = grid.shape
        grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 5:
                    if c == 13:
                        grid[r, c] = 9
                    elif c == 19:
                        grid[r, c] = 10
                    elif c == 20:
                        grid[r, c] = 10
                    elif c == 14:
                        grid[r, c] = 5
        return grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        grid = grid.copy()
        grid[py, px] = 15
        return grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    return grid[63, 0] == 15