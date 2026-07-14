import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 1 is a move down.
    # The game is empty, so no changes occur.
    return grid.copy()

def is_level_complete(grid):
    # The game is empty, so it is never complete.
    return False