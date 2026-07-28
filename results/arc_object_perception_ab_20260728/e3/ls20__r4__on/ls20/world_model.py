import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move all objects of color 5 one step down
        for r in range(H - 1):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r + 1, c] = 5
                    new_grid[r, c] = 0
    elif action == 2:
        # Move all objects of color 5 one step up
        for r in range(1, H):
            for c in range(W):
                if grid[r, c] == 5:
                    new_grid[r - 1, c] = 5
                    new_grid[r, c] = 0
    elif action == 3:
        # Move all objects of color 5 one step left
        for r in range(H):
            for c in range(1, W):
                if grid[r, c] == 5:
                    new_grid[r, c - 1] = 5
                    new_grid[r, c] = 0
    elif action == 4:
        # Move all objects of color 5 one step right
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] == 5:
                    new_grid[r, c + 1] = 5
                    new_grid[r, c] = 0
    elif action == 5:
        # Move all objects of color 5 one step down-left
        for r in range(H - 1):
            for c in range(1, W):
                if grid[r, c] == 5:
                    new_grid[r + 1, c - 1] = 5
                    new_grid[r, c] = 0
    elif action == 6:
        # Move all objects of color 5 one step down-right
        for r in range(H - 1):
            for c in range(W - 1):
                if grid[r, c] == 5:
                    new_grid[r + 1, c + 1] = 5
                    new_grid[r, c] = 0
    elif action == 7:
        # Move all objects of color 5 one step up-left
        for r in range(1, H):
            for c in range(1, W):
                if grid[r, c] == 5:
                    new_grid[r - 1, c - 1] = 5
                    new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all color 5 objects are in the bottom-right corner
    # This is a simplified check based on the win state pattern
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5:
                if r < H - 1 or c < W - 1:
                    return False
    return True