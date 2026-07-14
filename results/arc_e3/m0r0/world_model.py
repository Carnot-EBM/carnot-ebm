import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        # Convert pixel coordinates to logical grid coordinates
        gx, gy = px // 1, py // 1
        # Check bounds
        if gx < 0 or gx >= grid.shape[1] or gy < 0 or gy >= grid.shape[0]:
            return grid
        # Check if the cell is 0 (empty)
        if grid[gy, gx] == 0:
            # Toggle the cell to 5
            grid_copy = grid.copy()
            grid_copy[gy, gx] = 5
            return grid_copy
        # If the cell is not 0, no change
        return grid
    else:
        # For other actions, no change is observed in the provided data
        return grid

def is_level_complete(grid):
    # Check if the grid contains the win state pattern
    # Based on the initial grid and transitions, the win state is not explicitly defined
    # However, we can check for the presence of specific patterns or conditions
    # Since the win state is not explicitly defined, we return False
    return False