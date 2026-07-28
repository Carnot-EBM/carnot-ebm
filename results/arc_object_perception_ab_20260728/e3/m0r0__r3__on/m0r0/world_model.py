import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        r, c = py, px
        if grid[r, c] == 5:
            grid[r, c] = 0
        else:
            grid[r, c] = 5
        return grid
    elif action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        r, c = py, px
        if grid[r, c] == 5:
            grid[r, c] = 0
        else:
            grid[r, c] = 5
        return grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        r, c = py, px
        if grid[r, c] == 5:
            grid[r, c] = 0
        else:
            grid[r, c] = 5
        return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    return np.all(grid == 5)

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    return np.all(g == 0)
