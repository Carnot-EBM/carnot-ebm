import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if r == py and c == px:
                    new_grid[r, c] = 5
        return new_grid
    elif action == 5:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if r == py and c == px:
                    new_grid[r, c] = 0
        return new_grid
    elif action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(h):
            for c in range(w):
                if r == py and c == px:
                    new_grid[r, c] = 11
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        row_vals = grid[r, :]
        if len(np.unique(row_vals)) > 3:
            return False
        if np.sum(row_vals == 5) < 10:
            return False
    return True