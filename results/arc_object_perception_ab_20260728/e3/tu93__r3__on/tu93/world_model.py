import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        grid = grid.copy()
        grid[py, px] = 14
        return grid

    if action == 1:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    if c > 0 and grid[r, c-1] != 0:
                        grid[r, c] = grid[r, c-1]
                    elif r > 0 and grid[r-1, c] != 0:
                        grid[r, c] = grid[r-1, c]
        return grid

    if action == 2:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    if c > 0 and grid[r, c-1] != 0:
                        grid[r, c] = grid[r, c-1]
                    elif r > 0 and grid[r-1, c] != 0:
                        grid[r, c] = grid[r-1, c]
        return grid

    if action == 3:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    if c > 0 and grid[r, c-1] != 0:
                        grid[r, c] = grid[r, c-1]
                    elif r > 0 and grid[r-1, c] != 0:
                        grid[r, c] = grid[r-1, c]
        return grid

    if action == 4:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    if c > 0 and grid[r, c-1] != 0:
                        grid[r, c] = grid[r, c-1]
                    elif r > 0 and grid[r-1, c] != 0:
                        grid[r, c] = grid[r-1, c]
        return grid

    if action == 5:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    if c > 0 and grid[r, c-1] != 0:
                        grid[r, c] = grid[r, c-1]
                    elif r > 0 and grid[r-1, c] != 0:
                        grid[r, c] = grid[r-1, c]
        return grid

    if action == 7:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 0:
                    if c > 0 and grid[r, c-1] != 0:
                        grid[r, c] = grid[r, c-1]
                    elif r > 0 and grid[r-1, c] != 0:
                        grid[r, c] = grid[r-1, c]
        return grid

    return grid

def is_level_complete(grid):
    return np.all(grid == 5) or np.all(grid == 6) or np.all(grid == 14) or np.all(grid == 2) or np.all(grid == 0)