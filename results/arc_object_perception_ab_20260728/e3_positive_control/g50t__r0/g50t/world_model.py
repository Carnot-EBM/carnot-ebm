import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 9
        return grid
    
    if action == 2:
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 0:
                    grid[r, c] = 5
        return grid
    
    if action == 4:
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 0:
                    grid[r, c] = 9
        return grid
    
    return grid

def is_level_complete(grid):
    return np.all(grid == 0) or np.all(grid == 5) or np.all(grid == 9) or np.all(grid == 8)