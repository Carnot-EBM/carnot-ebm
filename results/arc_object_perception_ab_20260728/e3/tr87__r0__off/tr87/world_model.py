import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if r == py and c == px:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 1:
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    new_grid[r, c] = 5
        return new_grid
    elif action == 2:
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r, c] = 7
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 2 and grid[r, c] != 3 and grid[r, c] != 5 and grid[r, c] != 7:
                return False
    return True