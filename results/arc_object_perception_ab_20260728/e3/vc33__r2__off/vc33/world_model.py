import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y == 33:
            # Move left
            grid = grid.copy()
            if logical_x > 0:
                grid[logical_y, logical_x - 1] = grid[logical_y, logical_x]
                grid[logical_y, logical_x] = 4
            return grid
        elif logical_y == 34:
            # Move right
            grid = grid.copy()
            if logical_x < 63:
                grid[logical_y, logical_x + 1] = grid[logical_y, logical_x]
                grid[logical_y, logical_x] = 4
            return grid
        elif logical_y == 35:
            # Move up
            grid = grid.copy()
            if logical_y > 0:
                grid[logical_y - 1, logical_x] = grid[logical_y, logical_x]
                grid[logical_y, logical_x] = 4
            return grid
        elif logical_y == 36:
            # Move down
            grid = grid.copy()
            if logical_y < 63:
                grid[logical_y + 1, logical_x] = grid[logical_y, logical_x]
                grid[logical_y, logical_x] = 4
            return grid
        else:
            return grid.copy()
    return grid.copy()

def is_level_complete(grid):
    return np.array_equal(grid, np.array([
        [7] * 64,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [0] * 52 + [3] * 12,
        [9] * 4 + [0] * 48 + [3] * 12,
        [9] * 4 + [0] * 48 + [3] * 12,
        [9] * 4 + [0] * 48 + [3] * 12,
        [9] * 4 + [0] * 48 + [3] * 12,
        [5] * 56 + [3] * 8,
        [5] * 56 + [3] * 8,
        [5] * 56 + [3] * 8,
        [5] * 56 + [3] * 8,
        [9] * 4 + [0] * 8 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [0] * 12 + [3] * 52,
        [0] * 12 + [3] * 52,
        [0] * 12 + [3] * 52,
        [0] * 12 + [3] * 52,
        [0] * 12 + [3] * 52,
        [0] * 12 + [3] * 52,
        [0] * 12 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [9] * 4 + [0] * 8 + [3] * 52,
        [5] * 28 + [14] * 2 + [5] * 14 + [3] * 20,
        [5] * 28 + [14] * 2 + [5] * 14 + [3] * 20,
        [5] * 28 + [14] * 2 + [5] * 14 + [3] * 20,
        [5] * 28 + [14] * 2 + [5] * 14 + [3] * 20,
        [9] * 4 + [0] * 4 + [3] * 56,
        [9] * 4 + [0] * 4 + [3] * 56,
        [9] * 4 + [0] * 4 + [3] * 56,
        [9] * 4 + [0] * 4 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [14] * 2 + [4] * 2 + [3] * 52,
        [0] * 8 + [14] * 2 + [4] * 2 + [3] * 52,
        [0] * 8 + [14] * 2 + [4] * 4 + [3] * 50,
        [0] * 8 + [14] * 2 + [4] * 4 + [3] * 50,
        [0] * 8 + [14] * 2 + [4] * 2 + [3] * 52,
        [0] * 8 + [14] * 2 + [4] * 2 + [3] * 52,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56,
        [0] * 8 + [3] * 56
    ]))