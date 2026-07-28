import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 9
                    break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 9
                    break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 9
                    break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 9
                    break
    elif action == 5:
        # Toggle 9s to 10s
        new_grid = grid.copy()
        new_grid[grid == 9] = 10
    elif action == 6:
        # Click (no-op in this model)
        pass
    elif action == 7:
        # Toggle 10s to 9s
        new_grid = grid.copy()
        new_grid[grid == 10] = 9
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    target_row = 18
    target_col = 18
    
    if grid[target_row, target_col] != 9:
        return False
    
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 9 and grid[r, c] != 10 and grid[r, c] != 0 and grid[r, c] != 11:
                return False
                
    return True