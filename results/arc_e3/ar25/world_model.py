import numpy as np

def engine(grid, action, data):
    if action == 2:
        return apply_action_2(grid)
    elif action == 3:
        return apply_action_3(grid)
    elif action == 4:
        return apply_action_4(grid)
    elif action == 7:
        return grid
    elif action == 6:
        return grid
    else:
        return grid

def apply_action_2(grid):
    new_grid = grid.copy()
    h, w = grid.shape
    # Action 2: Move right
    # Identify all non-background cells (colors != 0)
    # Move them one step right if possible
    for r in range(h):
        for c in range(w - 1, 0, -1):
            if grid[r, c] != 0:
                if grid[r, c + 1] == 0:
                    new_grid[r, c + 1] = grid[r, c]
                    new_grid[r, c] = 0
    return new_grid

def apply_action_3(grid):
    new_grid = grid.copy()
    h, w = grid.shape
    # Action 3: Move left
    # Identify all non-background cells (colors != 0)
    # Move them one step left if possible
    for r in range(h):
        for c in range(w - 1):
            if grid[r, c] != 0:
                if grid[r, c - 1] == 0:
                    new_grid[r, c - 1] = grid[r, c]
                    new_grid[r, c] = 0
    return new_grid

def apply_action_4(grid):
    new_grid = grid.copy()
    h, w = grid.shape
    # Action 4: Move down
    # Identify all non-background cells (colors != 0)
    # Move them one step down if possible
    for r in range(h - 1, 0, -1):
        for c in range(w):
            if grid[r, c] != 0:
                if grid[r + 1, c] == 0:
                    new_grid[r + 1, c] = grid[r, c]
                    new_grid[r, c] = 0
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, the win state is when all non-background cells are collected
    # or when the grid is in a specific configuration
    # For this game, we assume the win state is when all non-background cells are in a specific pattern
    # or when the grid is fully cleared
    # Since we don't have explicit win state data, we return False as a placeholder
    # In a real scenario, this would check for specific conditions
    return False