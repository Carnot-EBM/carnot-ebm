import numpy as np

import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        if data is None:
            # Movement logic inferred from context
            pass
        else:
            # Click action
            px, py = data['x'], data['y']
            grid[py, px] = 0
    elif action == 2:
        # Directional movement
        pass
    elif action == 3:
        # Directional movement
        pass
    elif action == 4:
        # Directional movement
        pass
    elif action == 5:
        # Directional movement
        pass
    elif action == 6:
        # Click action
            px, py = data['x'], data['y']
            grid[py, px] = 0
    elif action == 7:
        # Directional movement
        pass
    return grid

def is_level_complete(grid):
    # Check for win state
    return False

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (8, 8):
        return False
    return np.all(grid == 0)
