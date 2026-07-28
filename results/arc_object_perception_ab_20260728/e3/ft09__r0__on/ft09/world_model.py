import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
        color = 8
        for dy in range(6):
            for dx in range(6):
                if 0 <= py + dy < H and 0 <= px + dx < W:
                    new_grid[py + dy, px + dx] = color
        return new_grid
    else:
        return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for y in range(H):
        for x in range(W):
            if grid[y, x] != 4 and grid[y, x] != 9 and grid[y, x] != 12:
                return False
    return True