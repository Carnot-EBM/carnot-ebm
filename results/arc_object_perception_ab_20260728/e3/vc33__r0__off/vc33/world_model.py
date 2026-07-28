import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 6:
        if data is None:
            return new_grid
        
        px, py = data['x'], data['y']
        if px < 0 or px >= W or py < 0 or py >= H:
            return new_grid
            
        new_grid[py, px] = 7
        
        if py >= 32:
            for r in range(py, H):
                if new_grid[r, px] == 3:
                    new_grid[r, px] = 0
                    if r < 32:
                        new_grid[r, px] = 3
        else:
            for r in range(py, -1, -1):
                if new_grid[r, px] == 3:
                    new_grid[r, px] = 0
                    if r > 32:
                        new_grid[r, px] = 3
                        
        return new_grid
    else:
        return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        for c in range(W):
            if grid[r, c] != 0 and grid[r, c] != 3 and grid[r, c] != 7:
                return False
    return True