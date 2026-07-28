import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    for dr in range(1, H):
                        if new_grid[r - dr, c] == 0:
                            new_grid[r - dr, c] = 5
                        else:
                            break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    for dr in range(1, H):
                        if new_grid[r + dr, c] == 0:
                            new_grid[r + dr, c] = 5
                        else:
                            break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    for dc in range(1, W):
                        if new_grid[r, c - dc] == 0:
                            new_grid[r, c - dc] = 5
                        else:
                            break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    for dc in range(1, W):
                        if new_grid[r, c + dc] == 0:
                            new_grid[r, c + dc] = 5
                        else:
                            break
    elif action == 5:
        # Toggle 0/11
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = 11
                elif new_grid[r, c] == 11:
                    new_grid[r, c] = 0
    elif action == 6:
        # Click (no-op in this context)
        pass
    elif action == 7:
        # Move Diagonal (Up-Right)
        for r in range(H - 1, -1, -1):
            for c in range(W):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    if r > 0 and c < W - 1 and new_grid[r - 1, c + 1] == 0:
                        new_grid[r - 1, c + 1] = 5
    elif action == 8:
        # Move Diagonal (Down-Left)
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    if r < H - 1 and c > 0 and new_grid[r + 1, c - 1] == 0:
                        new_grid[r + 1, c - 1] = 5
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if all 5s are at the top (row 0)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5:
                if r != 0:
                    return False
    # Check if all 0s are at the bottom (row H-1)
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0:
                if r != H - 1:
                    return False
    return True