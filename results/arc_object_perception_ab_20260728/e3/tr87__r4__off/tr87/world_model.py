import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move right
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] != 2 and grid[r, c + 1] == 2:
                    new_grid[r, c] = 2
                    new_grid[r, c + 1] = grid[r, c]
        return new_grid
    
    elif action == 2:
        # Move left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 2 and grid[r, c - 1] == 2:
                    new_grid[r, c] = 2
                    new_grid[r, c - 1] = grid[r, c]
        return new_grid
    
    elif action == 3:
        # Move down
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 2 and grid[r - 1, c] == 2:
                    new_grid[r, c] = 2
                    new_grid[r - 1, c] = grid[r, c]
        return new_grid
    
    elif action == 4:
        # Move up
        for c in range(W):
            for r in range(H):
                if grid[r, c] != 2 and grid[r + 1, c] == 2:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = grid[r, c]
        return new_grid
    
    elif action == 5:
        # Toggle 2 to 3
        new_grid = grid.copy()
        new_grid[grid == 2] = 3
        return new_grid
    
    elif action == 6:
        # Click action - no change
        return grid
    
    elif action == 7:
        # Toggle 3 to 2
        new_grid = grid.copy()
        new_grid[grid == 3] = 2
        return new_grid
    
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 2s are converted to 3s
    return np.all(grid[grid == 2] == 3)