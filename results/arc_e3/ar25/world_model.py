import numpy as np

def engine(grid, action, data):
    if action == 6:
        return grid.copy()
    
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 5
    elif action == 2:
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 10
    elif action == 3:
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 11
    elif action == 4:
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 4
    elif action == 5:
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 0
    elif action == 7:
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 9:
                    new_grid[r, c] = 10
    
    return new_grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                return False
    return True