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
        if action == 6:
            for dy in range(-2, 3):
                y = py + dy
                if y < 0 or y >= H:
                    continue
                for dx in range(-2, 3):
                    x = px + dx
                    if x < 0 or x >= W:
                        continue
                    if grid[y, x] == 12:
                        new_grid[y, x] = 0
                    else:
                        new_grid[y, x] = 15
    elif action == 7:
        if data is None:
            return new_grid
        # Action 7 does nothing
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 15:
                return False
    return True