import numpy as np

def engine(grid, action, data):
    if action == 3:
        # Action 3: Click on a specific cell to toggle or modify it
        if data is None:
            return grid
        
        # Parse the action data to find the clicked cell
        # The action data is in the format {'x': px, 'y': py}
        # Convert pixel coordinates to logical coordinates
        px, py = data['x'], data['y']
        row, col = py // 1, px // 1
        
        # Check if the cell is within bounds
        if row >= grid.shape[0] or col >= grid.shape[1]:
            return grid
        
        # Toggle the cell value
        # Based on the observed transitions, action 3 toggles the cell value
        # The toggle pattern is: 5 -> 9 -> 11 -> 5
        current_val = grid[row, col]
        if current_val == 5:
            grid[row, col] = 9
        elif current_val == 9:
            grid[row, col] = 11
        elif current_val == 11:
            grid[row, col] = 5
        elif current_val == 10:
            grid[row, col] = 10  # No change for 10
        else:
            grid[row, col] = 5  # Default to 5 for other values
        
        return grid
    
    elif action == 6:
        # Action 6: Click on a specific cell to place a value
        if data is None:
            return grid
        
        # Parse the action data to find the clicked cell
        px, py = data['x'], data['y']
        row, col = py // 1, px // 1
        
        # Check if the cell is within bounds
        if row >= grid.shape[0] or col >= grid.shape[1]:
            return grid
        
        # Place the value 15 in the clicked cell
        grid[row, col] = 15
        
        return grid
    
    else:
        # For other actions, return the grid unchanged
        return grid

def is_level_complete(grid):
    # Check if the level is complete
    # Based on the observed transitions, the level is complete when the last row (row 63)
    # is filled with the value 15
    if grid.shape[0] != 64 or grid.shape[1] != 64:
        return False
    
    last_row = grid[63, :]
    return np.all(last_row == 15)