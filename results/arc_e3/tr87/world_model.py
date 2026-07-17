import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                x, y = px + dx, py + dy
                if 0 <= x < W and 0 <= y < H:
                    if grid[y, x] == 0:
                        new_grid[y, x] = 3
        return new_grid
    elif action == 1:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 0:
                    new_grid[y, x] = 7
        return new_grid
    elif action == 2:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 0:
                    new_grid[y, x] = 5
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for y in range(H):
        for x in range(W):
            if grid[y, x] == 0:
                return False
    return True