import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Move up
        return move_direction(grid, -1)
    elif action == 2:
        # Action 2: Move down
        return move_direction(grid, 1)
    elif action == 3:
        # Action 3: Move left
        return move_direction(grid, 0, -1)
    elif action == 4:
        # Action 4: Move right
        return move_direction(grid, 0, 1)
    elif action == 5:
        # Action 5: Toggle (no-op in this context)
        return grid
    elif action == 6:
        # Action 6: Click (no-op in this context)
        return grid
    elif action == 7:
        # Action 7: No-op
        return grid
    return grid

def move_direction(grid, dr, dc=0):
    if dr == -1:
        return move_up(grid)
    elif dr == 1:
        return move_down(grid)
    elif dc == -1:
        return move_left(grid)
    elif dc == 1:
        return move_right(grid)
    return grid

def move_up(grid):
    H, W = grid.shape
    new_grid = grid.copy()
    # Shift everything up by 1
    new_grid[1:, :] = grid[:-1, :].copy()
    # Fill top row with 0s
    new_grid[0, :] = 0
    return new_grid

def move_down(grid):
    H, W = grid.shape
    new_grid = grid.copy()
    # Shift everything down by 1
    new_grid[:-1, :] = grid[1:, :].copy()
    # Fill bottom row with 0s
    new_grid[-1, :] = 0
    return new_grid

def move_left(grid):
    H, W = grid.shape
    new_grid = grid.copy()
    # Shift everything left by 1
    new_grid[:, 1:] = grid[:, :-1].copy()
    # Fill rightmost column with 0s
    new_grid[:, -1] = 0
    return new_grid

def move_right(grid):
    H, W = grid.shape
    new_grid = grid.copy()
    # Shift everything right by 1
    new_grid[:, :-1] = grid[:, 1:].copy()
    # Fill leftmost column with 0s
    new_grid[:, 0] = 0
    return new_grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Based on the observed transitions, the win state is when the grid is filled with 0s
    # or when the grid matches a specific pattern.
    # Since the observed transitions show the grid being filled with 0s,
    # we assume the win state is when the grid is all 0s.
    return np.all(grid == 0)