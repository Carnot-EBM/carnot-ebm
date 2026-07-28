import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move right
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] == 5 and new_grid[r, c + 1] != 5:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 5
    elif action == 2:
        # Move left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if new_grid[r, c] == 5 and new_grid[r, c - 1] != 5:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 5
    elif action == 3:
        # Move down
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] == 5 and new_grid[r - 1, c] != 5:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
    elif action == 4:
        # Move up
        for c in range(W):
            for r in range(H - 1):
                if new_grid[r, c] == 5 and new_grid[r + 1, c] != 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 5
    elif action == 6:
        # Click action - toggle cell at data coordinates
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if new_grid[py, px] == 5:
                new_grid[py, px] = 0
            else:
                new_grid[py, px] = 5
                
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 5s are in the rightmost column
    for r in range(H):
        for c in range(W - 1):
            if grid[r, c] == 5:
                return False
    # Check if there are any 5s at all
    if not np.any(grid == 5):
        return False
    return True