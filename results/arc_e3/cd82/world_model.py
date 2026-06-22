import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the current grid, action, and optional data.
    Implements the observed behavior:
    - Actions 1, 2: Change the bottom-right-most cell to 5.
    - Action 6: Click (no effect in this game).
    - Other actions: No effect.
    """
    grid = grid.copy()
    if action == 1:
        if data is not None:
            # Click action
            r, c = data['y'] - 1, data['x'] - 1
            grid[r, c] = 5
        else:
            # Keyboard action: change bottom-right cell
            grid[-1, -1] = 5
    elif action == 2:
        if data is not None:
            # Click action
            r, c = data['y'] - 1, data['x'] - 1
            grid[r, c] = 5
        else:
            # Keyboard action: change bottom-right cell
            grid[-1, -1] = 5
    elif action == 6:
        # Click action (no effect)
        pass
    return grid

def is_level_complete(grid):
    """
    Checks if the grid represents a completed level.
    Based on the observed data, the level is never completed in the provided examples.
    """
    return False