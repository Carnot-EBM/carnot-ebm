import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(w):
            for r in range(h - 1, 0, -1):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r - 1, c]
                    new_grid[r - 1, c] = 0
    elif action == 2:
        # Move Down
        for c in range(w):
            for r in range(h):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r + 1, c]
                    new_grid[r + 1, c] = 0
    elif action == 3:
        # Move Left
        for r in range(h):
            for c in range(w - 1, 0, -1):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r, c - 1]
                    new_grid[r, c - 1] = 0
    elif action == 4:
        # Move Right
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = new_grid[r, c + 1]
                    new_grid[r, c + 1] = 0
    elif action == 5:
        # Toggle
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 0:
                    new_grid[r, c] = 1
    elif action == 6:
        # Click
        if data:
            px, py = data['x'], data['y']
            r, c = py, px
            if 0 <= r < h and 0 <= c < w:
                new_grid[r, c] = 1
    elif action == 7:
        # Click
        if data:
            px, py = data['x'], data['y']
            r, c = py, px
            if 0 <= r < h and 0 <= c < w:
                new_grid[r, c] = 1
    
    return new_grid

def is_level_complete(grid):
    return np.all(grid == 0)