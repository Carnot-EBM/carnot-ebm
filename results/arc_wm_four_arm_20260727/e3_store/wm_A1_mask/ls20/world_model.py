import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1: Move Up
        for c in range(W):
            for r in range(H - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for rr in range(r - 1, -1, -1):
                        if new_grid[rr, c] != 0:
                            new_grid[r, c] = new_grid[rr, c]
                            new_grid[rr, c] = 0
                            break
        return new_grid
    
    elif action == 2:
        # Action 2: Move Right
        for r in range(H):
            for c in range(W - 1, -1, -1):
                if new_grid[r, c] == 0:
                    for cc in range(c - 1, -1, -1):
                        if new_grid[r, cc] != 0:
                            new_grid[r, c] = new_grid[r, cc]
                            new_grid[r, cc] = 0
                            break
        return new_grid
    
    elif action == 3:
        # Action 3: Move Down
        for c in range(W):
            for r in range(H):
                if new_grid[r, c] == 0:
                    for rr in range(r + 1, H):
                        if new_grid[rr, c] != 0:
                            new_grid[r, c] = new_grid[rr, c]
                            new_grid[rr, c] = 0
                            break
        return new_grid
    
    elif action == 4:
        # Action 4: Move Left
        for r in range(H):
            for c in range(W):
                if new_grid[r, c] == 0:
                    for cc in range(c + 1, W):
                        if new_grid[r, cc] != 0:
                            new_grid[r, c] = new_grid[r, cc]
                            new_grid[r, cc] = 0
                            break
        return new_grid
    
    elif action == 5:
        # Action 5: Toggle 0/1
        new_grid = grid.copy()
        new_grid[0, 0] = 1 - new_grid[0, 0]
        return new_grid
    
    elif action == 6:
        # Action 6: Click (handled by data)
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        new_grid[py, px] = 1 - new_grid[py, px]
        return new_grid
    
    elif action == 7:
        # Action 7: Toggle 0/1 at specific location (handled by data if needed, but data=None)
        new_grid = grid.copy()
        return new_grid
    
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    # Check if the grid is full of non-zero values
    return np.all(grid != 0)