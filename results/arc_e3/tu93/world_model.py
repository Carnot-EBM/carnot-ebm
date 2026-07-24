import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        row, col = py, px
        if 0 <= row < H and 0 <= col < W:
            if grid[row, col] == 5:
                grid[row, col] = 6
            elif grid[row, col] == 6:
                grid[row, col] = 5
    elif action == 2:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 0
    elif action == 3:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 2
    elif action == 4:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 9
    elif action == 1:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 14
    elif action == 5:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 4
    elif action == 7:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 15
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    if H != 64 or W != 64:
        return False
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5:
                return False
    return True