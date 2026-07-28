import numpy as np

def engine(grid, action, data):
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        grid[py, px] = 15
        return grid

    if action == 3:
        if data is None:
            return grid
        # Determine direction from action value
        # Actions 1-7 map to directions, but here we assume 3 is a specific action
        # Based on observed transitions, action 3 seems to be a directional move
        # We'll infer the direction from the pattern of changes
        # For simplicity, we'll assume action 3 is a specific move
        # In the observed data, action 3 changes cells in a specific pattern
        # We'll implement a simple logic based on the observed changes
        # This is a placeholder for the actual logic
        return grid

    return grid

def is_level_complete(grid):
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the grid
    # We'll check for the presence of specific patterns
    # This is a placeholder for the actual logic
    return False