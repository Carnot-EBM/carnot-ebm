import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
        grid[py, px] = 14
        return grid

    if action == 2:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 0
        return grid

    if action == 3:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 2
        return grid

    if action == 4:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 9
        return grid

    if action == 5:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 14
        return grid

    if action == 1:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 6
        return grid

    if action == 7:
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 5:
                    grid[r, c] = 0
        return grid

    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5:
                return False
    return True