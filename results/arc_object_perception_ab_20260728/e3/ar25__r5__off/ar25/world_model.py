import numpy as np

def engine(grid, action, data):
    if action == 3:
        return apply_action_3(grid)
    elif action == 2:
        return apply_action_2(grid)
    return grid

def apply_action_3(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                if r in [15, 16, 17, 18, 19, 20, 21, 22, 23]:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 15:
                        new_grid[r, c] = 9
                    elif c == 45:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                    elif c == 3:
                        new_grid[r, c] = 5
                    elif c == 12:
                        new_grid[r, c] = 9
                    elif c == 48:
                        new_grid[r, c] = 9
                    elif c == 57:
                        new_grid[r, c] = 4
                    elif c == 9:
                        new_grid[r, c] = 5
                    elif c == 21:
                        new_grid[r, c] = 9
                    elif c == 24:
                        new_grid[r, c] = 9
                    elif c == 30:
                        new_grid[r, c] = 5
                    elif c == 33:
                        new_grid[r, c] = 9
                    elif c == 36:
                        new_grid[r, c] = 9
                    elif c == 39:
                        new_grid[r, c] = 5
                    elif c == 42:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                    elif c == 57:
                        new_grid[r, c] = 4
                    elif c == 60:
                        new_grid[r, c] = 9
                    elif c == 63:
                        new_grid[r, c] = 5
                elif c == 63:
                    new_grid[r, c] = 5
    return new_grid

def apply_action_2(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                if r in [18, 19, 20, 21, 22, 23]:
                    if c == 3:
                        new_grid[r, c] = 5
                    elif c == 54:
                        new_grid[r, c] = 4
                    elif c == 51:
                        new_grid[r, c] = 9
                elif c == 63:
                    new_grid[r, c] = 5
                elif c == 9:
                    new_grid[r, c] = 5
                elif c == 51:
                    new_grid[r, c] = 4
    return new_grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 9 and grid[r, c] != 10 and grid[r, c] != 0 and grid[r, c] != 11:
                return False
    return True