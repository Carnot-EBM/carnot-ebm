import numpy as np

def engine(grid, action, data):
    if action == 1:
        return _move_left(grid)
    elif action == 2:
        return _move_right(grid)
    elif action == 3:
        return _move_up(grid)
    elif action == 4:
        return _move_down(grid)
    elif action == 5:
        return _move_left(grid)
    elif action == 6:
        return _move_right(grid)
    elif action == 7:
        return _move_up(grid)
    return grid

def _move_left(grid):
    return grid[:, :-1]

def _move_right(grid):
    return grid[:, 1:]

def _move_up(grid):
    return grid[1:, :]

def _move_down(grid):
    return grid[:-1, :]

def is_level_complete(grid):
    return np.all(grid == 2)