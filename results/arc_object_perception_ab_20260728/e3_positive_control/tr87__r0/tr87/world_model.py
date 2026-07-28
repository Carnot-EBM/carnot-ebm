import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 10
    elif action == 2:
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 10
    elif action == 3:
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 10
    elif action == 4:
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 10
    elif action == 5:
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 10
    elif action == 6:
        px, py = data['x'], data['y']
        new_grid[py, px] = 2
    elif action == 7:
        for r in range(H - 1):
            for c in range(W):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 2
                    new_grid[r + 1, c] = 10
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 10:
                return False
    return True