import numpy as np

def engine(grid, action, data):
    h, w = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        # Move right
        for r in range(h):
            for c in range(w - 1):
                if grid[r, c] != 0 and grid[r, c + 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r, c + 1] = grid[r, c]
                    break
    elif action == 2:
        # Move down
        for c in range(w):
            for r in range(h - 1):
                if grid[r, c] != 0 and grid[r + 1, c] == 0:
                    new_grid[r, c] = 0
                    new_grid[r + 1, c] = grid[r, c]
                    break
    elif action == 3:
        # Move left
        for r in range(h):
            for c in range(1, w):
                if grid[r, c] != 0 and grid[r, c - 1] == 0:
                    new_grid[r, c] = 0
                    new_grid[r, c - 1] = grid[r, c]
                    break
    elif action == 4:
        # Move up
        for c in range(w):
            for r in range(1, h):
                if grid[r, c] != 0 and grid[r - 1, c] == 0:
                    new_grid[r, c] = 0
                    new_grid[r - 1, c] = grid[r, c]
                    break
    elif action == 6:
        # Click action - toggle cell at data['x'], data['y']
        if data:
            px, py = data['x'], data['y']
            if 0 <= py < h and 0 <= px < w:
                new_grid[py, px] = 12 if grid[py, px] == 0 else 0
    elif action in [5, 7]:
        # Placeholder for other actions
        pass
    
    return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, the win state is when the grid is fully filled
    # or when specific conditions are met.
    # From the initial grid, row 63 is all 15s, which might indicate a win state.
    # However, the initial grid is not a win state.
    # The win state is likely when all non-zero cells are collected or arranged in a specific pattern.
    # Given the lack of explicit win state in the transitions, we assume the win state is when
    # the grid is fully filled with non-zero values or when a specific condition is met.
    # For simplicity, we check if the grid is fully filled with non-zero values.
    return np.all(grid != 0)