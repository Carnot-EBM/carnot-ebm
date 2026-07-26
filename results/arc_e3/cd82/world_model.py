import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 5
        return grid
    if action == 5:
        # Fill a vertical column with 15s
        col = data['x']
        for r in range(H):
            if grid[r, col] != 5:
                grid[r, col] = 15
        return grid
    if action == 3:
        # Fill a diagonal line of 15s from (data['x'], data['y'])
        x0, y0 = data['x'], data['y']
        for i in range(64):
            x, y = x0 + i, y0 + i
            if 0 <= x < W and 0 <= y < H:
                if grid[y, x] != 5:
                    grid[y, x] = 15
        return grid
    if action == 2:
        # Fill a diagonal line of 5s from (data['x'], data['y'])
        x0, y0 = data['x'], data['y']
        for i in range(64):
            x, y = x0 + i, y0 + i
            if 0 <= x < W and 0 <= y < H:
                if grid[y, x] != 5:
                    grid[y, x] = 5
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid is full of 5s
    return np.all(grid == 5)