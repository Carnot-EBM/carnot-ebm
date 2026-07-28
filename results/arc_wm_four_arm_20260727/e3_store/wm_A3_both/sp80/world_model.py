import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 15
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 15
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 15
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 15
    elif action == 5:
        # Move Up-Left
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c - 1] = 15
    elif action == 6:
        # Move Up-Right
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c + 1] = 15
    elif action == 7:
        # Move Down-Left
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c - 1] = 15
    
    if action == 6 and data:
        px, py = data['x'], data['y']
        new_grid[py // 1, px // 1] = 0
        new_grid[py // 1 - 1, px // 1 - 1] = 15
    
    return new_grid

def is_level_complete(grid):
    return np.all(grid == 15)