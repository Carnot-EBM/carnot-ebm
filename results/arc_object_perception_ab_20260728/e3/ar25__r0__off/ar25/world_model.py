import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = 9
                    break
    elif action == 2:
        # Move Down
        for c in range(W):
            for r in range(H):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = 9
                    break
    elif action == 3:
        # Move Left
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = 9
                    break
    elif action == 4:
        # Move Right
        for r in range(H):
            for c in range(W):
                if grid[r, c] == 9:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = 9
                    break
    elif action == 5:
        # Toggle 9 <-> 11
        new_grid = new_grid.copy()
        new_grid[new_grid == 9] = 11
        new_grid[new_grid == 11] = 9
    elif action == 6:
        # Click
        if data is not None:
            px, py = data['x'], data['y']
            logical_r, logical_c = py, px
            if 0 <= logical_r < H and 0 <= logical_c < W:
                if new_grid[logical_r, logical_c] == 9:
                    new_grid[logical_r, logical_c] = 11
                elif new_grid[logical_r, logical_c] == 11:
                    new_grid[logical_r, logical_c] = 9
    elif action == 7:
        # Toggle 9 <-> 10
        new_grid = new_grid.copy()
        new_grid[new_grid == 9] = 10
        new_grid[new_grid == 10] = 9
        
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        row_str = ""
        for c in range(W):
            row_str += str(grid[r, c])
        if row_str != "9" * 36 + "10" * 3 + "9" * 24 + "11" * 1:
            return False
    return True