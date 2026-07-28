import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 5:
                    if grid[r - 1, c] == 5:
                        new_grid[r, c] = 5
                        new_grid[r - 1, c] = grid[r, c]
                        grid[r, c] = 5
                        grid[r - 1, c] = grid[r, c]
        return new_grid
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 5:
                    if grid[r + 1, c] == 5:
                        new_grid[r, c] = 5
                        new_grid[r + 1, c] = grid[r, c]
                        grid[r, c] = 5
                        grid[r + 1, c] = grid[r, c]
        return new_grid
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 5:
                    if grid[r, c - 1] == 5:
                        new_grid[r, c] = 5
                        new_grid[r, c - 1] = grid[r, c]
                        grid[r, c] = 5
                        grid[r, c - 1] = grid[r, c]
        return new_grid
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    if grid[r, c + 1] == 5:
                        new_grid[r, c] = 5
                        new_grid[r, c + 1] = grid[r, c]
                        grid[r, c] = 5
                        grid[r, c + 1] = grid[r, c]
        return new_grid
    elif action == 5:
        # Move Up (with gravity)
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 5:
                    if grid[r - 1, c] == 5:
                        new_grid[r, c] = 5
                        new_grid[r - 1, c] = grid[r, c]
                        grid[r, c] = 5
                        grid[r - 1, c] = grid[r, c]
        return new_grid
    elif action == 6:
        # Click action (no-op in this context)
        return new_grid
    elif action == 7:
        # Move Down (with gravity)
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 5:
                    if grid[r + 1, c] == 5:
                        new_grid[r, c] = 5
                        new_grid[r + 1, c] = grid[r, c]
                        grid[r, c] = 5
                        grid[r, c + 1] = grid[r, c]
        return new_grid
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        row_str = ""
        for c in range(W):
            if c > 0 and grid[r, c] == grid[r, c - 1]:
                row_str += str(grid[r, c])
            else:
                row_str += str(grid[r, c])
        if row_str != "5x16,4x2,3x46":
            return False
    return True