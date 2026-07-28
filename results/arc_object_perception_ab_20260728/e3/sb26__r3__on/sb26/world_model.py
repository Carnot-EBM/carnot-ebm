import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px, py
        if logical_x >= grid.shape[1] or logical_y >= grid.shape[0]:
            return grid
        if grid[logical_y, logical_x] == 0:
            return grid
        target_color = grid[logical_y, logical_x]
        grid[logical_y, logical_x] = 0
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = logical_y + dy, logical_x + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    if grid[ny, nx] == target_color:
                        grid[ny, nx] = 0
        return grid
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    if grid.shape != (64, 64):
        return False
    if np.any(grid != 0):
        return False
    return True