import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        # Determine direction based on position relative to center (32, 32)
        if px < 32:
            direction = 1  # Right
        elif px > 32:
            direction = -1  # Left
        else:
            direction = 0  # Center
        
        # Apply movement logic based on direction
        if direction == 1:
            # Move right
            grid = grid.copy()
            grid[py, px:] = grid[py, px-1:px]
            grid[py, px-1] = 0
        elif direction == -1:
            # Move left
            grid = grid.copy()
            grid[py, :px] = grid[py, px+1:px+1+63-px]
            grid[py, px+1:] = 0
        return grid
    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # Win state has specific color patterns in rows
    # Simplified check: check if row 7 has the specific pattern
    if grid[7, 5] == 5 and grid[7, 6] == 5 and grid[7, 7] == 5 and grid[7, 8] == 5 and grid[7, 9] == 5:
        return True
    return False