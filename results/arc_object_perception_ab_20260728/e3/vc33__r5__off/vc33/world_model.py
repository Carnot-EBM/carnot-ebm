import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        
        # Identify the target cell (logical coordinate)
        tx, ty = px // 1, py // 1
        
        # Check if the target cell is within bounds
        if 0 <= ty < h and 0 <= tx < w:
            # Apply the transformation based on the observed pattern
            # The pattern shows a toggle or fill operation at the clicked position
            # Based on the observed transitions, clicking at (px, py) affects a specific region
            
            # Determine the affected region based on the click position
            # The observed data suggests a toggle or fill operation
            
            # Apply the transformation
            # This is a simplified version based on the observed pattern
            # The actual transformation might be more complex
            
            # Toggle the cell at the clicked position
            new_grid[ty, tx] = 7 if new_grid[ty, tx] != 7 else 0
            
            # Apply additional transformations based on the observed pattern
            # This is a simplified version based on the observed pattern
            
            return new_grid
        else:
            return grid
    else:
        # For other actions, return the grid unchanged
        return grid

def is_level_complete(grid):
    h, w = grid.shape
    # Check if the grid matches the win state pattern
    # The win state has specific patterns in the grid
    
    # Check the top row
    if not np.all(grid[0] == 7):
        return False
    
    # Check the bottom row
    if not np.all(grid[-1] == 8):
        return False
    
    # Check the middle rows for specific patterns
    for i in range(1, h-1):
        if not np.all(grid[i] == 0):
            return False
    
    return True