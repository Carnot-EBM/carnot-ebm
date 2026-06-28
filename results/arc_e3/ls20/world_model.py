import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= grid.shape[0] or py < 0 or py >= grid.shape[1]:
            return grid
        if grid[px, py] == 0:
            return grid
        target_row = px - 1
        target_col = py - 1
        if target_row < 0 or target_col < 0:
            return grid
        if grid[target_row, target_col] == 0:
            return grid
        if grid[target_row, target_col] == grid[px, py]:
            return grid
        new_grid = grid.copy()
        new_grid[px, py] = 0
        new_grid[target_row, target_col] = grid[px, py]
        return new_grid
    return grid

def is_level_complete(grid):
    return False

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (21, 21):
        return False
    if grid.shape[0] != grid.shape[1]:
        return False
    return np.all(grid == 0)
