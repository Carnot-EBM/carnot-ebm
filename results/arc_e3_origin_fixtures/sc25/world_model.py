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
            for r in range(H - 1):
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
            for c in range(W - 1):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r, c + 1]
                    new_grid[r, c + 1] = 0
    elif action == 6:
        # Click action - no effect in this model
        pass
    
    return new_grid

def is_level_complete(grid):
    # Check if all non-zero cells are in the bottom-right region
    # Based on the win state pattern observed
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0:
                if r < H - 1 or c < W - 1:
                    return False
    return True