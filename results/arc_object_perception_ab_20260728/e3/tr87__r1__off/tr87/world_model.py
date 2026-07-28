import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move right
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 2
                    new_grid[r, c + 1] = 5
    elif action == 2:
        # Move down
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 5
    elif action == 3:
        # Move left
        for r in range(H):
            for c in range(1, W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 2
                    new_grid[r, c - 1] = 5
    elif action == 4:
        # Move up
        for r in range(1, H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 2
                    new_grid[r - 1, c] = 5
    elif action == 5:
        # Toggle 5 <-> 7
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 7
                elif new_grid[r, c] == 7:
                    new_grid[r, c] = 5
    elif action == 6:
        # Click (no-op in this model)
        pass
    elif action == 7:
        # Move all 5s to the rightmost available 2s
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 2
                    new_grid[r, c + 1] = 5
    elif action == 8:
        # Move all 5s to the leftmost available 2s
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 2
                    new_grid[r, c - 1] = 5
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 5s have been converted to 2s
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5:
                return False
    return True