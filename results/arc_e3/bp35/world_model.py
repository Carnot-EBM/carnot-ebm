import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        if 0 <= py < grid.shape[0] and 0 <= px < grid.shape[1]:
            grid[py, px] = 15
    elif action == 4:
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 10:
                    if c + 1 < w and grid[r, c + 1] == 0:
                        grid[r, c + 1] = 15
    elif action == 3:
        h, w = grid.shape
        for r in range(h):
            for c in range(w):
                if grid[r, c] == 10:
                    if c - 1 >= 0 and grid[r, c - 1] == 0:
                        grid[r, c - 1] = 15
    return grid

def is_level_complete(grid):
    return np.all(grid == 0)