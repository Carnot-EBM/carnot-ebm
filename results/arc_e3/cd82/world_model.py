import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, the game is static with no changes.
    grid_copy = grid.copy()
    return grid_copy

def is_level_complete(grid):
    # return True if `grid` is a level-complete / win state, else False.
    # Based on observed transitions, the level is never complete.
    return False