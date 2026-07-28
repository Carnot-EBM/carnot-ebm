import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 3:
                    new_grid[r, c] = 0
                    for dr in range(1, H):
                        if new_grid[r - dr, c] == 0:
                            new_grid[r - dr, c] = 3
                            break
                        elif new_grid[r - dr, c] != 0:
                            new_grid[r - dr, c] = 3
                            break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 3:
                    new_grid[r, c] = 0
                    for dr in range(1, H + 1):
                        if r + dr < H and new_grid[r + dr, c] == 0:
                            new_grid[r + dr, c] = 3
                            break
                        elif r + dr < H and new_grid[r + dr, c] != 0:
                            new_grid[r + dr, c] = 3
                            break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 3:
                    new_grid[r, c] = 0
                    for dc in range(1, W):
                        if new_grid[r, c - dc] == 0:
                            new_grid[r, c - dc] = 3
                            break
                        elif new_grid[r, c - dc] != 0:
                            new_grid[r, c - dc] = 3
                            break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 3:
                    new_grid[r, c] = 0
                    for dc in range(1, W + 1):
                        if c + dc < W and new_grid[r, c + dc] == 0:
                            new_grid[r, c + dc] = 3
                            break
                        elif c + dc < W and new_grid[r, c + dc] != 0:
                            new_grid[r, c + dc] = 3
                            break
    elif action == 5:
        # Toggle corners
        new_grid[0, W - 1] = 0
        new_grid[H - 1, 0] = 0
    elif action == 6:
        # Click (no-op in this model)
        pass
    elif action == 7:
        # No-op
        pass
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 3s are collected (only 4s and 14s remain)
    return np.all(grid != 3)