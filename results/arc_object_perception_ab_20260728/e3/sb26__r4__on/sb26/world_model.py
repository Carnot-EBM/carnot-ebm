import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 16 or logical_y > 47:
            return grid
        if logical_x < 17 or logical_x > 46:
            return grid
        if grid[logical_y, logical_x] != 4:
            return grid
        new_grid = grid.copy()
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = logical_y + dy, logical_x + dx
                if 0 <= ny < 64 and 0 <= nx < 64:
                    if grid[ny, nx] == 4:
                        new_grid[ny, nx] = 0
        return new_grid
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    if grid.shape != (64, 64):
        return False
    for y in range(64):
        for x in range(64):
            if grid[y, x] != 4 and grid[y, x] != 5:
                return False
    for y in range(64):
        row = grid[y, :]
        if y < 8 or y > 7:
            if row[0] != 4 or row[1] != 4 or row[2] != 4 or row[3] != 4:
                return False
            if row[63] != 4 or row[62] != 4 or row[61] != 4 or row[60] != 4:
                return False
        else:
            if row[0] != 4 or row[1] != 4 or row[2] != 4 or row[3] != 4:
                return False
            if row[63] != 4 or row[62] != 4 or row[61] != 4 or row[60] != 4:
                return False
    return True