import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 0:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 5
        return new_grid
    elif action == 1:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 4
        return new_grid
    elif action == 2:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 14
        return new_grid
    elif action == 3:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 3
        return new_grid
    elif action == 4:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 1
        return new_grid
    elif action == 5:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 15
        return new_grid
    elif action == 6:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 11
        return new_grid
    elif action == 7:
        if data is None:
            return grid
        if data['x'] is None:
            return grid
        row = data['y']
        col = data['x']
        if row < 0 or row >= grid.shape[0] or col < 0 or col >= grid.shape[1]:
            return grid
        new_grid = grid.copy()
        new_grid[row, col] = 9
        return new_grid
    return grid

def is_level_complete(grid):
    return True

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.shape[0] != 10 or g.shape[1] != 10:
        return False
    return np.all(g == 0)
