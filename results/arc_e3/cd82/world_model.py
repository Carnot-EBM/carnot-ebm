import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on the observed transitions (empty grid, no changes), the engine returns the grid unchanged.
    return grid.copy()

def is_level_complete(grid):
    # Based on the observed transitions (empty grid, no changes), the level is never complete.
    return False