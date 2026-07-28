import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 5
        return grid
    
    if action == 2:
        for step in range(10):
            new_grid = grid.copy()
            for r in range(H):
                for c in range(W):
                    if grid[r, c] == 0:
                        for dr in range(1, 10):
                            if r + dr < H and c + dr < W and grid[r + dr, c] != 0:
                                new_grid[r, c] = grid[r + dr, c]
                                break
            if np.array_equal(grid, new_grid):
                break
            grid = new_grid
        return grid
    
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0:
                return False
    return True