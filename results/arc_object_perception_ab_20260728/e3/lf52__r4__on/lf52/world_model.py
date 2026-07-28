import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        new_grid = grid.copy()
        r = 3
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx == 0 and dy == 0:
                    continue
                ny, nx = py + dy, px + dx
                if 0 <= ny < H and 0 <= nx < W:
                    if grid[ny, nx] == 0:
                        new_grid[ny, nx] = 5
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for y in range(H):
        for x in range(W):
            if grid[y, x] == 0:
                return False
    return True