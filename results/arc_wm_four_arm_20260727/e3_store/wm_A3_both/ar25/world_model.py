import numpy as np

def engine(grid, action, data):
    if action == 7:
        return grid
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] == 5:
            return grid
        return grid.copy()
    if action == 2:
        return apply_action_2(grid)
    if action == 3:
        return apply_action_3(grid)
    if action == 4:
        return apply_action_4(grid)
    return grid

def apply_action_2(grid):
    new_grid = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 11:
                new_grid[r, c] = 14
    return new_grid

def apply_action_3(grid):
    new_grid = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 11:
                new_grid[r, c] = 14
    return new_grid

def apply_action_4(grid):
    new_grid = grid.copy()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 11:
                new_grid[r, c] = 14
    return new_grid

def is_level_complete(grid):
    return False