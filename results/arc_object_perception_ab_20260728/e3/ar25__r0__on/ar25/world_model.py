import numpy as np

def engine(grid, action, data):
    if action == 3:
        return apply_action_3(grid)
    elif action == 2:
        return apply_action_2(grid)
    else:
        return grid

def apply_action_3(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if new_grid[r, c] == 9:
                new_grid[r, c] = 5
    return new_grid

def apply_action_2(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if new_grid[r, c] == 9:
                new_grid[r, c] = 5
    return new_grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                return False
    return True