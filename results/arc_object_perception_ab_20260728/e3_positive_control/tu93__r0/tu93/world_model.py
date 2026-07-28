import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, 0, -1):
                if grid[r, c] != 5:
                    if grid[r - 1, c] == 5:
                        new_grid[r, c] = 5
                        new_grid[r - 1, c] = grid[r, c]
                        break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H - 1):
                if grid[r, c] != 5:
                    if grid[r + 1, c] == 5:
                        new_grid[r, c] = 5
                        new_grid[r + 1, c] = grid[r, c]
                        break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, 0, -1):
                if grid[r, c] != 5:
                    if grid[r, c - 1] == 5:
                        new_grid[r, c] = 5
                        new_grid[r, c - 1] = grid[r, c]
                        break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] != 5:
                    if grid[r, c + 1] == 5:
                        new_grid[r, c] = 5
                        new_grid[r, c + 1] = grid[r, c]
                        break
    elif action == 6:
        px, py = data['x'], data['y']
        if 0 <= py < H and 0 <= px < W:
            if grid[py, px] != 5:
                new_grid[py, px] = 5
    elif action == 7:
        # Toggle
        if data is not None:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                if grid[py, px] == 5:
                    new_grid[py, px] = 0
                else:
                    new_grid[py, px] = 5
        else:
            # Toggle all 5s to 0
            new_grid = grid.copy()
            new_grid[new_grid == 5] = 0
            
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 5:
                return False
    return True