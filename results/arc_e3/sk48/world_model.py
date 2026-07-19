import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 3:
        if data is None:
            return grid
        # Action 3 toggles a 6x6 block at (data['x'], data['y'])
        x, y = data['x'], data['y']
        grid[y:y+6, x:x+6] = 0
    elif action == 6:
        if data is None:
            return grid
        # Action 6 is a click at (data['x'], data['y'])
        x, y = data['x'], data['y']
        grid[y, x] = 0
    return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, the win state is when the grid is all zeros
    return np.all(grid == 0)