import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 10:
                    if grid[r - 1, c] == 10:
                        new_grid[r, c] = new_grid[r - 1, c]
                        new_grid[r - 1, c] = grid[r, c]
                        break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 10:
                    if grid[r + 1, c] == 10:
                        new_grid[r, c] = new_grid[r + 1, c]
                        new_grid[r + 1, c] = grid[r, c]
                        break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 10:
                    if grid[r, c - 1] == 10:
                        new_grid[r, c] = new_grid[r, c - 1]
                        new_grid[r, c - 1] = grid[r, c]
                        break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W - 1):
                if grid[r, c] != 10:
                    if grid[r, c + 1] == 10:
                        new_grid[r, c] = new_grid[r, c + 1]
                        new_grid[r, c + 1] = grid[r, c]
                        break
    elif action == 5:
        # Toggle 0 <-> 10
        new_grid = new_grid.copy()
        new_grid[new_grid == 0] = 10
        new_grid[new_grid == 10] = 0
    elif action == 6:
        # Click
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            ly, lx = py, px
            if 0 <= ly < H and 0 <= lx < W:
                new_grid[ly, lx] = 12
    elif action == 7:
        # Toggle 10 <-> 12
        new_grid = new_grid.copy()
        new_grid[new_grid == 10] = 12
        new_grid[new_grid == 12] = 10
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 12:
                return False
    return True

import numpy as np

def is_level_complete(grid):
    grid = np.array(grid)
    if grid.shape[0] == 0 or grid.shape[1] == 0:
        return False
    return np.all(grid == 0)
