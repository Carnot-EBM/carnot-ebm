import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(w):
            for r in range(h - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 5
                    break
    elif action == 2:
        # Action 2: Move Down
        for c in range(w):
            for r in range(h):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 5
                    break
    elif action == 3:
        # Action 3: Move Left
        for r in range(h):
            for c in range(w - 1, -1, -1):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 5
                    break
    elif action == 4:
        # Action 4: Move Right
        for r in range(h):
            for c in range(w):
                if new_grid[r, c] == 5:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 5
                    break
    elif action == 6:
        # Action 6: Click (toggle)
        if data:
            px, py = data['x'], data['y']
            if 0 <= py < h and 0 <= px < w:
                new_grid[py, px] = 1 - new_grid[py, px]
    elif action == 7:
        # Action 7: Toggle all 5s to 0
        new_grid[new_grid == 5] = 0
    
    return new_grid

def is_level_complete(grid):
    return np.all(grid == 0)