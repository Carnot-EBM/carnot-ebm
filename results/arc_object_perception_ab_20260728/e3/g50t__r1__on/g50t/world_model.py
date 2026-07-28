import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_x < 0 or logical_x >= grid.shape[1] or logical_y < 0 or logical_y >= grid.shape[0]:
            return grid
        grid = grid.copy()
        grid[logical_y, logical_x] = 5
        return grid
    return grid

def is_level_complete(grid):
    return False

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape[0] < 2 or grid.shape[1] < 2:
        return False
    return np.all(grid[1:] == grid[:-1]) and np.all(grid[:, 1:] == grid[:, :-1])
