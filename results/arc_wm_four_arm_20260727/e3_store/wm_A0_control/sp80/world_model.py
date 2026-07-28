import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r - 1, c]
                    new_grid[r - 1, c] = 15
    elif action == 2:
        # Move down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r + 1, c]
                    new_grid[r + 1, c] = 15
    elif action == 3:
        # Move left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r, c - 1]
                    new_grid[r, c - 1] = 15
    elif action == 4:
        # Move right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r, c + 1]
                    new_grid[r, c + 1] = 15
    elif action == 5:
        # Move up-left
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r - 1, c]
                    new_grid[r - 1, c] = 15
    elif action == 6:
        # Move up-right
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r - 1, c]
                    new_grid[r - 1, c] = 15
    elif action == 7:
        # Move down-right
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] == 15:
                    new_grid[r, c] = grid[r + 1, c]
                    new_grid[r + 1, c] = 15
    
    return new_grid

def is_level_complete(grid):
    return False