import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 15
    elif action == 3:
        if data is None:
            return new_grid
        # Determine direction from action code (simplified mapping)
        # Action 3 is Down
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = 15
    return new_grid

def is_level_complete(grid):
    return False