import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] != 5:
                    new_grid[r, c] = 5
                    break
        return new_grid
    
    elif action == 2:
        # Action 2: Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] != 5:
                    new_grid[r, c] = 5
                    break
        return new_grid
    
    elif action == 3:
        # Action 3: Move Left
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] != 5:
                    new_grid[r, c] = 5
                    break
        return new_grid
    
    elif action == 4:
        # Action 4: Move Right
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] != 5:
                    new_grid[r, c] = 5
                    break
        return new_grid
    
    elif action == 6:
        # Action 6: Click
        if data and 'x' in data and 'y' in data:
            px, py = data['x'], data['y']
            if 0 <= py < H and 0 <= px < W:
                new_grid[py, px] = 0
        return new_grid
    
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    return grid[0, 0] == 0