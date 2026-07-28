import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move right
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] == 5 and grid[r, c + 1] != 2:
                    new_grid[r, c] = grid[r, c + 1]
                    new_grid[r, c + 1] = 5
    elif action == 2:
        # Move left
        for r in range(H):
            for c in range(1, W):
                if grid[r, c] == 5 and grid[r, c - 1] != 2:
                    new_grid[r, c] = grid[r, c - 1]
                    new_grid[r, c - 1] = 5
    elif action == 3:
        # Move down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] == 5 and grid[r + 1, c] != 2:
                    new_grid[r, c] = grid[r + 1, c]
                    new_grid[r + 1, c] = 5
    elif action == 4:
        # Move up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 5 and grid[r - 1, c] != 2:
                    new_grid[r, c] = grid[r - 1, c]
                    new_grid[r - 1, c] = 5
    elif action == 6:
        # Click action - toggle specific cell
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if new_grid[py, px] == 5:
                new_grid[py, px] = 2
            else:
                new_grid[py, px] = 5
    elif action == 7:
        # Toggle all 5s to 2s
        new_grid[new_grid == 5] = 2
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 5s are converted to 2s
    return np.all(grid[grid == 5] == 2)