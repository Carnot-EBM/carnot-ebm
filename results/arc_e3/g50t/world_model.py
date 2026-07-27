import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Right
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] != 0 and new_grid[r, c + 1] == 0:
                    new_grid[r, c + 1] = new_grid[r, c]
                    new_grid[r, c] = 0
    elif action == 2:
        # Move Down
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] != 0 and new_grid[r + 1, c] == 0:
                    new_grid[r + 1, c] = new_grid[r, c]
                    new_grid[r, c] = 0
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(1, W):
                if new_grid[r, c] != 0 and new_grid[r, c - 1] == 0:
                    new_grid[r, c - 1] = new_grid[r, c]
                    new_grid[r, c] = 0
    elif action == 4:
        # Move Up
        for r in range(1, H):
            for c in range(W):
                if new_grid[r, c] != 0 and new_grid[r - 1, c] == 0:
                    new_grid[r - 1, c] = new_grid[r, c]
                    new_grid[r, c] = 0
    elif action == 6:
        # Click (no-op in this model)
        pass
    elif action == 7:
        # Toggle (no-op in this model)
        pass
        
    return new_grid

def is_level_complete(grid):
    # Check if the grid is full of 15s (win state)
    return np.all(grid == 15)