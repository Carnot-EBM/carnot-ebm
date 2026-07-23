import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, the game state remains unchanged after any action.
    # The initial grid is all zeros, and no changes occur.
    return grid.copy()

def is_level_complete(grid):
    # Based on observed transitions, the level is never complete.
    return False