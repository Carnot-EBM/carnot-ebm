import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 3:
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(45, 50):
            new_grid[r, 29:34] = 5
            new_grid[r, 45:50] = 5
        new_grid[61, 13:18] = 1
        new_grid[62, 13:18] = 1
        return new_grid
    elif action == 2:
        h, w = grid.shape
        new_grid = grid.copy()
        for r in range(61, 63):
            for c in range(14, 19):
                new_grid[r, c] = 1
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    return False

def is_level_complete(grid):
    import numpy as np
    grid = np.asarray(grid)
    if grid.shape != (10, 10):
        return False
    if grid.dtype != object:
        return False
    if not np.all([isinstance(c, str) for c in grid.flat]):
        return False
    if not np.all([len(c) == 1 for c in grid.flat]):
        return False
    if not np.all([c in '0123456789' for c in grid.flat]):
        return False
    if np.all(grid == '0'):
        return True
    return False
