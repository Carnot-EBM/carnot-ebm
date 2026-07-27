import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 0
        return new_grid
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if new_grid[r, c] != 1:
                    new_grid[r, c] = new_grid[r - 1, c]
                    new_grid[r - 1, c] = 1
        return new_grid
    
    if action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] != 1:
                    new_grid[r, c] = new_grid[r + 1, c]
                    new_grid[r + 1, c] = 1
        return new_grid
    
    if action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if new_grid[r, c] != 1:
                    new_grid[r, c] = new_grid[r, c - 1]
                    new_grid[r, c - 1] = 1
        return new_grid
    
    if action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] != 1:
                    new_grid[r, c] = new_grid[r, c + 1]
                    new_grid[r, c + 1] = 1
        return new_grid
    
    if action == 5:
        # Toggle
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 1:
                    new_grid[r, c] = 0
        return new_grid
    
    if action == 7:
        # Clear
        new_grid[:] = 0
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    return grid[63, 63] == 0