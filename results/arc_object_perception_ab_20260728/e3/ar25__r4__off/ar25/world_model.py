import numpy as np

def engine(grid, action, data):
    if action == 3:
        return apply_action_3(grid)
    elif action == 2:
        return apply_action_2(grid)
    else:
        return grid

def apply_action_3(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                if r >= 15 and r <= 23:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 6:
                        new_grid[r, c] = 5
                    elif c == 15:
                        new_grid[r, c] = 9
                    elif c == 45:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
    return new_grid

def apply_action_2(grid):
    h, w = grid.shape
    new_grid = grid.copy()
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                if r >= 18 and r <= 23:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 24 and r <= 29:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 30 and r <= 35:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 36 and r <= 41:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 42 and r <= 47:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 48 and r <= 53:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 54 and r <= 59:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
                elif r >= 60 and r <= 63:
                    if c == 63:
                        new_grid[r, c] = 5
                    elif c == 3:
                        new_grid[r, c] = 9
                    elif c == 51:
                        new_grid[r, c] = 9
                    elif c == 54:
                        new_grid[r, c] = 4
    return new_grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        if r % 2 == 0:
            if not (grid[r, 0:36] == 9).all() or not (grid[r, 36:39] == 10).all() or not (grid[r, 39:63] == 9).all() or grid[r, 63] != 11:
                return False
        else:
            if not (grid[r, 0:36] == 9).all() or not (grid[r, 36:39] == 10).all() or not (grid[r, 39:63] == 9).all() or grid[r, 63] != 11:
                return False
    return True