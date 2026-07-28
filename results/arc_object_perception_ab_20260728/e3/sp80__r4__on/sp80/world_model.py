import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 4:
        if data is None:
            return new_grid
        px, py = data['x'], data['y']
        # Determine the row and column based on pixel coordinates
        # The grid is 64x64, and pixel coordinates are logical*1
        # We need to find the row and column that corresponds to the pixel
        # Since the grid is 64x64, we can assume the pixel coordinates are directly mapped to the grid
        row = py // 1
        col = px // 1
        # Apply the action to the grid
        # The action is to toggle the cell at (row, col)
        new_grid[row, col] = 1 - new_grid[row, col]
        return new_grid
    
    elif action == 5:
        # This action is more complex and involves multiple changes
        # Based on the observed transitions, this action seems to fill the grid with specific patterns
        # We will implement a simplified version that matches the observed behavior
        # The action seems to fill the grid with a specific pattern based on the initial state
        # We will use a heuristic to determine the new state
        # This is a simplified version and may not cover all cases
        # The actual implementation would require more detailed analysis of the observed transitions
        # For now, we will return the grid as is
        return new_grid
    
    else:
        # For other actions, return the grid as is
        return new_grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed win state, we can check for specific patterns
    # The win state has a specific structure that we can check for
    # We will check if the grid matches the win state pattern
    # This is a simplified version and may not cover all cases
    # The actual implementation would require more detailed analysis of the observed win state
    # For now, we will return False
    return False