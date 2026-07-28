import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        new_grid = grid.copy()
        new_grid[logical_y, logical_x] = 7
        return new_grid
    return grid

def is_level_complete(grid):
    return np.all(grid == grid[0])