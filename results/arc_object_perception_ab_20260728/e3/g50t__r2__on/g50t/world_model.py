import numpy as np

def engine(grid, action, data):
    if action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 0 or logical_y >= 64 or logical_x < 0 or logical_x >= 64:
            return grid
        grid = grid.copy()
        grid[logical_y, logical_x] = 5
        return grid
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    grid = np.array(grid)
    if grid.shape != (64, 64):
        return False
    return bool(np.all(grid == grid[0, 0]))