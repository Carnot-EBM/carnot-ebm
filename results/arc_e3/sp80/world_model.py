import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 1:
                    if r - 1 >= 0 and grid[r - 1, c] == 0:
                        new_grid[r, c] = 0
                        new_grid[r - 1, c] = 1
                    elif r - 1 >= 0 and grid[r - 1, c] == 1:
                        new_grid[r, c] = 0
                        new_grid[r - 1, c] = 1
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] == 1:
                    if r + 1 < H and grid[r + 1, c] == 0:
                        new_grid[r, c] = 0
                        new_grid[r + 1, c] = 1
                    elif r + 1 < H and grid[r + 1, c] == 1:
                        new_grid[r, c] = 0
                        new_grid[r + 1, c] = 1
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 1:
                    if c - 1 >= 0 and grid[r, c - 1] == 0:
                        new_grid[r, c] = 0
                        new_grid[r, c - 1] = 1
                    elif c - 1 >= 0 and grid[r, c - 1] == 1:
                        new_grid[r, c] = 0
                        new_grid[r, c - 1] = 1
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 1:
                    if c + 1 < W and grid[r, c + 1] == 0:
                        new_grid[r, c] = 0
                        new_grid[r, c + 1] = 1
                    elif c + 1 < W and grid[r, c + 1] == 1:
                        new_grid[r, c] = 0
                        new_grid[r, c + 1] = 1
    elif action == 5:
        # Toggle specific cell
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                if new_grid[py, px] == 0:
                    new_grid[py, px] = 1
                else:
                    new_grid[py, px] = 0
    elif action == 6:
        # Click with data
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                if new_grid[py, px] == 0:
                    new_grid[py, px] = 1
                else:
                    new_grid[py, px] = 0
    elif action == 7:
        # Toggle all
        new_grid[:] = 1 - new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all cells are filled with 1
    return np.all(grid == 1)