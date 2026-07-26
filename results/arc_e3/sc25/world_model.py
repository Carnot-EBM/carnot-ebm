import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if c < W - 1:
                        new_grid[r, c + 1] = 5
                        new_grid[r, c] = 0
    elif action == 2:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if r < H - 1:
                        new_grid[r + 1, c] = 5
                        new_grid[r, c] = 0
    elif action == 3:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if c < W - 1:
                        new_grid[r, c + 1] = 5
                        new_grid[r, c] = 0
                    elif r < H - 1:
                        new_grid[r + 1, c] = 5
                        new_grid[r, c] = 0
    elif action == 4:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if c < W - 1:
                        new_grid[r, c + 1] = 5
                        new_grid[r, c] = 0
                    elif r < H - 1:
                        new_grid[r + 1, c] = 5
                        new_grid[r, c] = 0
    elif action == 5:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if c < W - 1:
                        new_grid[r, c + 1] = 5
                        new_grid[r, c] = 0
                    elif r < H - 1:
                        new_grid[r + 1, c] = 5
                        new_grid[r, c] = 0
    elif action == 6:
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 5
    elif action == 7:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    if c < W - 1:
                        new_grid[r, c + 1] = 5
                        new_grid[r, c] = 0
                    elif r < H - 1:
                        new_grid[r + 1, c] = 5
                        new_grid[r, c] = 0
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 5:
                return False
    return True