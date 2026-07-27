import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        # Convert pixel to logical
        r, c = py // 1, px // 1
        if r >= grid.shape[0] or c >= grid.shape[1]:
            return grid
        # Apply the observed pattern: set a 6x1 vertical line at (r, c) to 0,
        # and set two 1x1 cells at (r+1, c+5) and (r+2, c+5) to 1.
        # This matches the observed deltas for action 6.
        grid[r, c:c+6] = 0
        if r + 1 < grid.shape[0] and c + 5 < grid.shape[1]:
            grid[r + 1, c + 5] = 1
        if r + 2 < grid.shape[0] and c + 5 < grid.shape[1]:
            grid[r + 2, c + 5] = 1
        return grid
    elif action == 7:
        return grid
    else:
        return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern observed in the initial grid
    # The win state is characterized by specific patterns in the lower rows.
    # Based on the initial grid, rows 53-63 have specific patterns.
    # We check if the grid matches the win state pattern.
    # The win state is identified by the presence of specific colors in the lower rows.
    # For simplicity, we check if the grid has the same pattern as the initial grid.
    # This is a placeholder for the actual win state check.
    # In the absence of a clear win state pattern, we return False.
    return False

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.shape != (10, 10):
        return False
    return np.all(g == 0)
