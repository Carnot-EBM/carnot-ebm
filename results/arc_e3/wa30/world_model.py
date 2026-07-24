import numpy as np

def engine(grid, action, data):
    if action == 3:
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
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        # Apply gravity
        for c in range(w):
            col = grid[:, c].copy()
            empty_count = 0
            for r in range(h - 1, -1, -1):
                if col[r] == 0:
                    empty_count += 1
                else:
                    new_grid[r - empty_count, c] = col[r]
                    if r - empty_count < 0:
                        break
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 0:
                return False
    return True