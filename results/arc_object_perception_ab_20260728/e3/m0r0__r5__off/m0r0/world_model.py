import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 1:
        if data is None:
            return new_grid
        
        # Action 1 is a click that toggles a specific cell
        # The data contains the coordinates of the cell to toggle
        # We need to find the cell that was toggled based on the delta
        
        # Since we don't have the exact coordinates in the data, we need to infer them
        # from the delta. The delta shows which cells changed.
        
        # For simplicity, we'll assume the action toggles a cell at a specific location
        # based on the pattern of changes
        
        # Let's look at the pattern of changes for action 1
        # It seems to toggle cells in a specific pattern
        
        # For now, let's just return the grid as is
        # This is a placeholder - we need to figure out the exact logic
        
        return new_grid
    
    elif action == 3:
        if data is None:
            return new_grid
        
        # Action 3 is a click that toggles a specific cell
        # Similar to action 1
        
        return new_grid
    
    # For other actions, we don't have enough information
    return new_grid

def is_level_complete(grid):
    H, W = grid.shape
    
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the rows
    
    # Let's check the first and last rows
    # In the win state, row 0 is all 5s, row 63 is all 5s
    # And there are specific patterns in the middle rows
    
    # For simplicity, let's check if the grid matches the win state exactly
    # We'll compare the run-length encoding
    
    # This is a placeholder - we need to figure out the exact logic
    return False