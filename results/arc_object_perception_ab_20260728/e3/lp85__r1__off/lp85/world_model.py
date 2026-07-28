import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r - 1, c]
                    new_grid[r - 1, c] = 0
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r + 1, c]
                    new_grid[r + 1, c] = 0
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r, c - 1]
                    new_grid[r, c - 1] = 0
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r, c + 1]
                    new_grid[r, c + 1] = 0
    elif action == 5:
        # Rotate Left
        new_grid = np.rot90(new_grid, k=1)
    elif action == 6:
        # Click
        px, py = data['x'], data['y']
        r, c = py // 1, px // 1
        new_grid[r, c] = 0
    elif action == 7:
        # Rotate Right
        new_grid = np.rot90(new_grid, k=-1)
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0:
                return False
    return True