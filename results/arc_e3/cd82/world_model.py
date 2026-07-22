import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on the provided transitions, the game state remains unchanged after the action.
    # Therefore, the engine simply returns the input grid.
    return grid

def is_level_complete(grid):
    # Based on the provided transitions, the win state is not reached in the observed data.
    # A default implementation checks if the grid is all zeros (empty), which is a common win condition.
    # However, without a clear win state definition from the data, this is a placeholder.
    # Given the strict instruction to induce rules from data and the data shows no change,
    # we return False.
    return False