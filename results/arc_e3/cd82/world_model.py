import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (8x8 int). Return the predicted next grid (same shape).
    # Action 1-7 are directional movements. Action 6 is a click.
    # Based on the initial grid being empty (all 0s) and no changes observed,
    # the world model is that the grid remains unchanged.
    return grid.copy()

def is_level_complete(grid):
    # The initial grid is empty. Without further observations of a win state,
    # we assume the level is not complete unless the grid is empty (start state).
    # However, typically a win state is distinct. Given no data, we return False.
    return False