import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Right
        for r in range(H):
            for c in range(W - 1):
                if new_grid[r, c] != 2 and new_grid[r, c + 1] == 2:
                    new_grid[r, c] = 2
                    new_grid[r, c + 1] = new_grid[r, c]
        return new_grid
    
    elif action == 2:
        # Move Down
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] != 2 and new_grid[r + 1, c] == 2:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = new_grid[r, c]
        return new_grid
    
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(1, W):
                if new_grid[r, c] != 2 and new_grid[r, c - 1] == 2:
                    new_grid[r, c] = 2
                    new_grid[r, c - 1] = new_grid[r, c]
        return new_grid
    
    elif action == 4:
        # Move Up
        for r in range(H - 1, 0, -1):
            for c in range(W):
                if new_grid[r, c] != 2 and new_grid[r - 1, c] == 2:
                    new_grid[r, c] = 2
                    new_grid[r - 1, c] = new_grid[r, c]
        return new_grid
    
    elif action == 6:
        # Click
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 3
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 2:
                return False
    return True