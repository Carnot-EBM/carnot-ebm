import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 14
        return grid
    
    if action == 0:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    if action == 1:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    if action == 2:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    if action == 3:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    if action == 4:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    if action == 5:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    if action == 7:
        grid = grid.copy()
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 3:
                    grid[r, c] = 14
        return grid
    
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] == 3:
                return False
    return True