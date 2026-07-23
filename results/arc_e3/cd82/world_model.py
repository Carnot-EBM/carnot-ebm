import numpy as np

import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    if action == 1:
        return grid
    return grid

def is_level_complete(grid):
    return False

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    if np.any(grid != 0):
        return False
    return True
