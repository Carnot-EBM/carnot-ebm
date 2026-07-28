import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    return False

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    return np.all(grid == 2)
