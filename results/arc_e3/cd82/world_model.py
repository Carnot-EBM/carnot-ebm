import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(w):
            for r in range(h - 1, -1, -1):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = grid[r, c]
                    break
    elif action == 2:
        # Action 2: Move Down
        for c in range(w):
            for r in range(h):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = grid[r, c]
                    break
    elif action == 3:
        # Action 3: Move Left
        for r in range(h):
            for c in range(w - 1, -1, -1):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = grid[r, c]
                    break
    elif action == 4:
        # Action 4: Move Right
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = grid[r, c]
                    break
    elif action == 5:
        # Action 5: Move Diagonal Up-Left
        for r in range(h - 1, -1, -1):
            for c in range(w - 1, -1, -1):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c - 1] = grid[r, c]
                    break
    elif action == 6:
        # Action 6: Move Diagonal Up-Right
        for r in range(h - 1, -1, -1):
            for c in range(w):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c + 1] = grid[r, c]
                    break
    elif action == 7:
        # Action 7: Move Diagonal Down-Left
        for r in range(h):
            for c in range(w - 1, -1, -1):
                if grid[r, c] != 0:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c - 1] = grid[r, c]
                    break
    
    return new_grid

def is_level_complete(grid):
    return False