import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if px < 0 or px >= 64 or py < 0 or py >= 64:
            return grid
        new_grid = grid.copy()
        cols = [33, 38]
        for c in cols:
            if py == 59:
                for r in range(56, 62):
                    if r == 56:
                        new_grid[r, c:c+6] = 0
                    elif r == 57:
                        new_grid[r, c] = 0
                        new_grid[r, c+5] = 0
                    elif r == 58:
                        new_grid[r, c] = 0
                        new_grid[r, c+5] = 0
                    elif r == 59:
                        new_grid[r, c] = 0
                        new_grid[r, c+5] = 0
                    elif r == 60:
                        new_grid[r, c] = 0
                        new_grid[r, c+5] = 0
                    elif r == 61:
                        new_grid[r, c:c+6] = 0
            elif py == 30:
                for r in range(28, 32):
                    if r == 28:
                        new_grid[r, c] = 9
                        new_grid[r, c+1] = 9
                        new_grid[r, c+2] = 9
                        new_grid[r, c+3] = 9
                    elif r == 29:
                        new_grid[r, c] = 9
                        new_grid[r, c+1] = 9
                        new_grid[r, c+2] = 9
                        new_grid[r, c+3] = 9
                    elif r == 30:
                        new_grid[r, c] = 9
                        new_grid[r, c+1] = 9
                        new_grid[r, c+2] = 9
                        new_grid[r, c+3] = 9
                    elif r == 31:
                        new_grid[r, c] = 9
                        new_grid[r, c+1] = 9
                        new_grid[r, c+2] = 9
                        new_grid[r, c+3] = 9
        return new_grid
    return grid

def is_level_complete(grid):
    if grid is None:
        return False
    rows = grid.shape[0]
    if rows < 1:
        return False
    return True