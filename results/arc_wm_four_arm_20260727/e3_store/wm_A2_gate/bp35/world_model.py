import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 7:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            new_grid[0, 0] = 1
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            new_grid[py, px] = 1
        else:
            new_grid[0, 0] = 1
    elif action == 4:
        # Action 4 toggles a 3x3 area centered at (0,0)
        for i in range(3):
            for j in range(3):
                if 0 <= i < H and 0 <= j < W:
                    new_grid[i, j] = 1
    elif action == 3:
        # Action 3 toggles a 3x3 area centered at (0,0)
        for i in range(3):
            for j in range(3):
                if 0 <= i < H and 0 <= j < W:
                    new_grid[i, j] = 1
    
    return new_grid

def is_level_complete(grid):
    return np.all(grid == 1)