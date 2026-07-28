import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        grid[py, px] = 7
        return grid
    elif action == 1:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 7:
                    if c < grid.shape[1] - 1:
                        grid[r, c + 1] = 7
                        grid[r, c] = 0
        return grid
    elif action == 2:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 7:
                    if r < grid.shape[0] - 1:
                        grid[r + 1, c] = 7
                        grid[r, c] = 0
        return grid
    elif action == 3:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 7:
                    grid[r, c] = 0
        return grid
    elif action == 4:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 7:
                    grid[r, c] = 1
        return grid
    elif action == 5:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 7:
                    grid[r, c] = 2
        return grid
    elif action == 7:
        grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 7:
                    grid[r, c] = 3
        return grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 0)