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
                    break
        return new_grid
    
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    break
        return new_grid
        
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    break
        return new_grid
        
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 15:
                    new_grid[r, c] = 0
                    break
        return new_grid
        
    elif action == 5:
        # Toggle 0/15 at (0, 60)
        new_grid[0, 60] = 1 - new_grid[0, 60]
        return new_grid
        
    elif action == 6:
        # Toggle 0/15 at (0, 58)
        new_grid[0, 58] = 1 - new_grid[0, 58]
        return new_grid
        
    elif action == 7:
        # Toggle 0/15 at (0, 55)
        new_grid[0, 55] = 1 - new_grid[0, 55]
        return new_grid
        
    return new_grid

def is_level_complete(grid):
    # Check if all 15s are collected (only 1s remain)
    return np.all(grid == 1)