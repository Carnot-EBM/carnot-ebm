import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        logical_x, logical_y = px // 1, py // 1
        if logical_y < 0 or logical_y >= H or logical_x < 0 or logical_x >= W:
            return grid
        new_grid = grid.copy()
        new_grid[logical_y, logical_x] = 0
        return new_grid
    elif action == 5:
        new_grid = grid.copy()
        new_grid[:, 0] = 1
        new_grid[4:8, 12::8] = 11
        new_grid[4:8, 28::8] = 11
        new_grid[4:8, 44::8] = 11
        new_grid[8:12, 12::8] = 11
        new_grid[8:12, 20::8] = 11
        new_grid[8:12, 28::8] = 11
        new_grid[8:12, 36::8] = 11
        new_grid[8:12, 44::8] = 11
        new_grid[8:12, 52::8] = 11
        new_grid[16:20, 8::8] = 8
        new_grid[16:20, 24::8] = 12
        new_grid[24:28, 28::8] = 8
        new_grid[36:40, 20::8] = 9
        new_grid[52:56, 16::8] = 12
        new_grid[52:56, 24::8] = 12
        new_grid[52:56, 40::8] = 12
        new_grid[52:56, 48::8] = 12
        new_grid[53:56, 16::8] = 12
        new_grid[53:56, 24::8] = 12
        new_grid[53:56, 40::8] = 12
        new_grid[53:56, 48::8] = 12
        new_grid[54:56, 16::8] = 12
        new_grid[54:56, 24::8] = 12
        new_grid[54:56, 40::8] = 12
        new_grid[54:56, 48::8] = 12
        new_grid[55:56, 16::8] = 12
        new_grid[55:56, 24::8] = 12
        new_grid[55:56, 40::8] = 12
        new_grid[55:56, 48::8] = 12
        new_grid[56:59, 16::8] = 12
        new_grid[56:59, 40::8] = 8
        new_grid[57:59, 16::8] = 12
        new_grid[57:59, 40::8] = 8
        new_grid[58:59, 16::8] = 12
        new_grid[58:59, 40::8] = 8
        new_grid[59:60, 16::8] = 12
        new_grid[59:60, 40::8] = 8
        new_grid[60:64, 0:40] = 12
        new_grid[60:64, 40:44] = 4
        new_grid[60:64, 44:64] = 12
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(64):
        if r < 4:
            if grid[r, :] != 1:
                return False
        elif r < 8:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 12:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 16:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 20:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 28:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 36:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 40:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 52:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 56:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 60:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        elif r < 63:
            if grid[r, 0:12] != 12 or grid[r, 12:24] != 11 or grid[r, 24:28] != 12 or grid[r, 28:40] != 11 or grid[r, 40:44] != 12 or grid[r, 44:48] != 11 or grid[r, 48:56] != 12:
                return False
        else:
            if grid[r, :] != 14:
                return False
    return True