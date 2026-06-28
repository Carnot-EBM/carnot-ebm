import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Action 1: Move left
        return move(grid, 0)
    elif action == 2:
        # Action 2: Move right
        return move(grid, 1)
    elif action == 3:
        # Action 3: Move up
        return move(grid, -1)
    elif action == 4:
        # Action 4: Move down
        return move(grid, 1)
    elif action == 5:
        # Action 5: Toggle color 0 to 10
        return toggle(grid, 0, 10)
    elif action == 6:
        # Action 6: Click (no-op in this model)
        return grid
    elif action == 7:
        # Action 7: Toggle color 0 to 11
        return toggle(grid, 0, 11)
    else:
        return grid

def move(grid, direction):
    # direction: 0=left, 1=right, -1=up, 1=down
    # This is a simplified movement model based on observed transitions
    # The observed transitions show color changes in specific regions
    # We simulate movement by shifting non-background colors
    new_grid = grid.copy()
    h, w = grid.shape
    
    if direction == 0:  # Left
        for r in range(h):
            for c in range(w - 1, -1, -1):
                if grid[r, c] != 0 and grid[r, c] != 5:
                    if c > 0 and grid[r, c - 1] == 0:
                        new_grid[r, c - 1] = grid[r, c]
                        new_grid[r, c] = 0
    elif direction == 1:  # Right
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 0 and grid[r, c] != 5:
                    if c < w - 1 and grid[r, c + 1] == 0:
                        new_grid[r, c + 1] = grid[r, c]
                        new_grid[r, c] = 0
    elif direction == -1:  # Up
        for c in range(w):
            for r in range(h - 1, -1, -1):
                if grid[r, c] != 0 and grid[r, c] != 5:
                    if r > 0 and grid[r - 1, c] == 0:
                        new_grid[r - 1, c] = grid[r, c]
                        new_grid[r, c] = 0
    elif direction == 1:  # Down
        for c in range(w):
            for r in range(h):
                if grid[r, c] != 0 and grid[r, c] != 5:
                    if r < h - 1 and grid[r + 1, c] == 0:
                        new_grid[r + 1, c] = grid[r, c]
                        new_grid[r, c] = 0
    
    return new_grid

def toggle(grid, from_color, to_color):
    # Toggle specific colors
    new_grid = grid.copy()
    new_grid[grid == from_color] = to_color
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on observed transitions, win state has specific color patterns
    # This is a simplified check
    h, w = grid.shape
    # Check if all non-background colors are collected
    # This is a heuristic based on the observed win state
    return np.all(grid == 0) or np.all(grid == 5) or np.all(grid == 10) or np.all(grid == 11) or np.all(grid == 12)