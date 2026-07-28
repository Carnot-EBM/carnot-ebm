import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if y == py and x == px:
                    new_grid[y, x] = 0
        return new_grid
    elif action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if y == py and x == px:
                    new_grid[y, x] = 0
        return new_grid
    elif action == 1:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 10:
                    new_grid[y, x] = 0
        return new_grid
    elif action == 2:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 5:
                    new_grid[y, x] = 7
        return new_grid
    elif action == 3:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 7:
                    new_grid[y, x] = 5
        return new_grid
    elif action == 5:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 2:
                    new_grid[y, x] = 3
        return new_grid
    elif action == 7:
        new_grid = grid.copy()
        for y in range(H):
            for x in range(W):
                if grid[y, x] == 3:
                    new_grid[y, x] = 2
        return new_grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for y in range(H):
        for x in range(W):
            if grid[y, x] == 10:
                return False
            if grid[y, x] == 2:
                return False
    return True