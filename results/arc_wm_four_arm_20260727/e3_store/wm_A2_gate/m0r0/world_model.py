import numpy as np

def engine(grid, action, data):
    if action == 2:
        # Action 2: Place a vertical line of color 5 at specific columns
        # Based on observed changes, it seems to place color 5 at columns 19 and 39
        # from rows 49 to 58
        grid = grid.copy()
        for row in range(49, 59):
            grid[row, 19] = 5
            grid[row, 39] = 5
        return grid
    elif action == 5:
        # Action 5: Place color 0 at corners
        grid = grid.copy()
        grid[0, 63] = 0
        grid[63, 0] = 0
        return grid
    elif action == 1:
        # Action 1: Place color 5 at specific positions and color 0 at (63, 1)
        grid = grid.copy()
        for row in range(49, 59):
            grid[row, 19] = 5
            grid[row, 39] = 5
        grid[63, 1] = 0
        return grid
    elif action == 4:
        # Action 4: Place color 5 at specific positions and color 0 at (63, 2)
        grid = grid.copy()
        for row in range(49, 59):
            grid[row, 19] = 5
            grid[row, 34] = 5
        grid[63, 2] = 0
        return grid
    elif action == 3:
        # Action 3: Place color 5 at specific positions
        grid = grid.copy()
        for row in range(49, 54):
            grid[row, 19] = 5
            grid[row, 34] = 5
        return grid
    elif action == 6:
        # Action 6: Click action - no change observed in data
        return grid
    else:
        # Default: no change
        return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Based on the initial grid and transitions, the win state seems to be when
    # the grid is fully filled with the initial pattern
    # Since we don't have explicit win state data, we check if the grid is unchanged
    # from the initial state (which would indicate no progress)
    # However, based on the game mechanics, the win state is likely when
    # all the colored blocks are in their final positions
    # For now, we'll return False as we don't have enough information to determine the win state
    return False