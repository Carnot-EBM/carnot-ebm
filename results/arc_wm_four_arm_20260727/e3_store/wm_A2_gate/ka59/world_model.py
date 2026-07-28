import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 0
    elif action == 4:
        # Action 4: Push down at column 18
        col = 18
        for r in range(H - 1, -1, -1):
            if new_grid[r, col] == 15:
                new_grid[r, col] = 10
                if r > 0:
                    new_grid[r - 1, col] = 15
    elif action == 1:
        # Action 1: Push down at column 21
        col = 21
        for r in range(H - 1, -1, -1):
            if new_grid[r, col] == 15:
                new_grid[r, col] = 10
                if r > 0:
                    new_grid[r - 1, col] = 15
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    if grid[63, 63] == 0:
        return True
    return False