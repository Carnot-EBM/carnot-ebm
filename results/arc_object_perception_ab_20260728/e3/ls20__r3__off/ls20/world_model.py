import numpy as np

def engine(grid, action, data):
    if action == 1:
        return move_down(grid)
    elif action == 2:
        return move_left(grid)
    elif action == 3:
        return move_right(grid)
    elif action == 4:
        return move_up(grid)
    elif action == 5:
        return move_down(grid)
    elif action == 6:
        return move_down(grid)
    elif action == 7:
        return move_left(grid)
    return grid

def move_down(grid):
    grid = grid.copy()
    H, W = grid.shape
    for c in range(W):
        for r in range(H - 1, 0, -1):
            if grid[r, c] == 0:
                grid[r, c] = grid[r - 1, c]
                grid[r - 1, c] = 0
    return grid

def move_left(grid):
    grid = grid.copy()
    H, W = grid.shape
    for r in range(H):
        for c in range(W - 1, 0, -1):
            if grid[r, c] == 0:
                grid[r, c] = grid[r, c - 1]
                grid[r, c - 1] = 0
    return grid

def move_right(grid):
    grid = grid.copy()
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 0:
                grid[r, c] = grid[r, c + 1]
                grid[r, c + 1] = 0
    return grid

def move_up(grid):
    grid = grid.copy()
    H, W = grid.shape
    for c in range(W):
        for r in range(H):
            if grid[r, c] == 0:
                grid[r, c] = grid[r + 1, c]
                grid[r + 1, c] = 0
    return grid

def is_level_complete(grid):
    return np.all(grid == 0)