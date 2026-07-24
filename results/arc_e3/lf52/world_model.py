import numpy as np

def engine(grid, action, data):
    """
    Applies a single action to the grid and returns the resulting grid.
    The game is a simple toggle mechanic:
    - Action 1: Toggle all cells in the grid (0->1, 1->0).
    - Action 2: Toggle all cells in the grid (0->1, 1->0).
    - Action 3: Toggle all cells in the grid (0->1, 1->0).
    - Action 4: Toggle all cells in the grid (0->1, 1->0).
    - Action 5: Toggle all cells in the grid (0->1, 1->0).
    - Action 6: Toggle the cell at the given pixel coordinates (x, y).
    - Action 7: Toggle all cells in the grid (0->1, 1->0).
    """
    grid = grid.copy()
    
    if action == 6:
        if data is not None:
            x = data['x']
            y = data['y']
            # Toggle the cell at (x, y)
            grid[y, x] = 1 - grid[y, x]
    else:
        # Toggle all cells in the grid
        grid = 1 - grid
        
    return grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    The level is complete if all cells in the grid are 1.
    """
    return np.all(grid == 1)