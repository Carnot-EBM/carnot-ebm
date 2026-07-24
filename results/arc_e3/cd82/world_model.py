import numpy as np

def engine(grid, action, data):
    if action == 1:
        return grid
    if action == 2:
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 5:
                    if c > 0 and grid[r, c-1] != 5:
                        grid[r, c-1] = 5
                        grid[r, c] = 0
        return grid
    if action == 3:
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 4:
                    if c < 63 and grid[r, c+1] != 4:
                        grid[r, c+1] = 4
                        grid[r, c] = 0
        return grid
    if action == 4:
        grid = grid.copy()
        if data:
            px, py = data['x'], data['y']
            grid[py, px] = 5
            grid[py, px-1] = 5
            grid[py-1, px] = 5
            grid[py-1, px-1] = 5
        return grid
    if action == 5:
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 3:
                    if r > 0 and grid[r-1, c] != 3:
                        grid[r-1, c] = 3
                        grid[r, c] = 0
        return grid
    if action == 6:
        return grid
    if action == 7:
        grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 0:
                    if r < 63 and grid[r+1, c] != 0:
                        grid[r+1, c] = 0
                        grid[r, c] = 5
        return grid
    return grid

def is_level_complete(grid):
    return grid[2, 32] == 4