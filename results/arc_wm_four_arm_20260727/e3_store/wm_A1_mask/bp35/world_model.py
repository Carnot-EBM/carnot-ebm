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
        # Action 4 toggles a 3x3 block centered at (H//2, W//2)
        cx, cy = H // 2, W // 2
        for i in range(-1, 2):
            for j in range(-1, 2):
                r, c = cy + i, cx + j
                if 0 <= r < H and 0 <= c < W:
                    new_grid[r, c] = 1
    elif action == 3:
        # Action 3 toggles a 3x3 block centered at (H//2, W//2)
        cx, cy = H // 2, W // 2
        for i in range(-1, 2):
            for j in range(-1, 2):
                r, c = cy + i, cx + j
                if 0 <= r < H and 0 <= c < W:
                    new_grid[r, c] = 1
    
    return new_grid

def is_level_complete(grid):
    return np.all(grid == 1)