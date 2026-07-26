import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid.copy()
    
    px, py = data['x'], data['y']
    h, w = grid.shape
    
    # Check if player is at (px, py)
    if grid[px, py] != 14:
        return grid.copy()
    
    # Create a copy of the grid
    new_grid = grid.copy()
    
    # Check if player is at the center of a 5x5 block
    if px % 5 == 2 and py % 5 == 2:
        # Check if the 5x5 block is complete (all 14s)
        if np.all(grid[px-2:px+3, py-2:py+3] == 14):
            # Check if the 5x5 block is not already filled with 5s
            if not np.all(grid[px-2:px+3, py-2:py+3] == 5):
                # Fill the 5x5 block with 5s
                new_grid[px-2:px+3, py-2:py+3] = 5
                # Remove the 14s from the grid
                new_grid[px-2:px+3, py-2:py+3] = 5
                return new_grid
    
    return grid.copy()

def is_level_complete(grid):
    h, w = grid.shape
    # Check if all 14s have been converted to 5s
    if np.any(grid == 14):
        return False
    
    # Check if the grid is filled with 5s
    if np.all(grid == 5):
        return True
    
    # Check if the grid is filled with 14s
    if np.all(grid == 14):
        return False
    
    return False