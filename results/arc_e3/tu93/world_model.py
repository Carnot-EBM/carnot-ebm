import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 1:
        # Move Up
        new_grid = grid.copy()
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] != 5:
                    target = r - 1
                    while target >= 0 and grid[target, c] == 5:
                        target -= 1
                    if target >= 0:
                        new_grid[target, c] = grid[r, c]
                        new_grid[r, c] = 5
        return new_grid
    elif action == 2:
        # Move Down
        new_grid = grid.copy()
        for c in range(W):
            for r in range(H):
                if grid[r, c] != 5:
                    target = r + 1
                    while target < H and grid[target, c] == 5:
                        target += 1
                    if target < H:
                        new_grid[target, c] = grid[r, c]
                        new_grid[r, c] = 5
        return new_grid
    elif action == 3:
        # Move Left
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] != 5:
                    target = c - 1
                    while target >= 0 and grid[r, target] == 5:
                        target -= 1
                    if target >= 0:
                        new_grid[r, target] = grid[r, c]
                        new_grid[r, c] = 5
        return new_grid
    elif action == 4:
        # Move Right
        new_grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    target = c + 1
                    while target < W and grid[r, target] == 5:
                        target += 1
                    if target < W:
                        new_grid[r, target] = grid[r, c]
                        new_grid[r, c] = 5
        return new_grid
    elif action == 6:
        # Click (data contains x, y)
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        if 0 <= py < H and 0 <= px < W:
            new_grid[py, px] = 0
        return new_grid
    else:
        return grid

def is_level_complete(grid):
    return grid[63, 63] == 0