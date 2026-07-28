import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] == 0 and new_grid[r - 1, c] != 0:
                    new_grid[r, c] = new_grid[r - 1, c]
                    new_grid[r - 1, c] = 0
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if new_grid[r, c] == 0 and new_grid[r + 1, c] != 0:
                    new_grid[r, c] = new_grid[r + 1, c]
                    new_grid[r + 1, c] = 0
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if new_grid[r, c] == 0 and new_grid[r, c - 1] != 0:
                    new_grid[r, c] = new_grid[r, c - 1]
                    new_grid[r, c - 1] = 0
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0 and new_grid[r, c + 1] != 0:
                    new_grid[r, c] = new_grid[r, c + 1]
                    new_grid[r, c + 1] = 0
    elif action == 5:
        # Toggle
        if data is not None:
            r, c = data['y'] // 1, data['x'] // 1
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 0 if new_grid[r, c] != 0 else 1
    elif action == 6:
        # Click (no-op in observed data)
        pass
    elif action == 7:
        # No-op
        pass
        
    return new_grid

def is_level_complete(grid):
    return False