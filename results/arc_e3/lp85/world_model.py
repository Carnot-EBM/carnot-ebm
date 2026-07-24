import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 0:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 1:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 2:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 3:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 4:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 5:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 6:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    elif action == 7:
        if data is None:
            return grid
        new_grid = grid.copy()
        for r in range(64):
            for c in range(64):
                if new_grid[r, c] == 14:
                    new_grid[r, c] = 3
        return new_grid
    return grid

def is_level_complete(grid):
    return True

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    return np.all(grid == 0)
