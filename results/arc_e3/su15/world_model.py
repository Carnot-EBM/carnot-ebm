import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if grid[py, px] != 0:
            return grid
        # Find the nearest non-zero value in the same row (left or right)
        row = py
        left_val = None
        right_val = None
        for c in range(px - 1, -1, -1):
            if grid[row, c] != 0:
                left_val = grid[row, c]
                break
        for c in range(px + 1, grid.shape[1]):
            if grid[row, c] != 0:
                right_val = grid[row, c]
                break
        
        # Determine the value to set
        if left_val is not None and right_val is not None:
            val = (left_val + right_val) // 2
        elif left_val is not None:
            val = left_val
        elif right_val is not None:
            val = right_val
        else:
            val = 0
        
        # Set the cell to the determined value
        grid[py, px] = val
        return grid.copy()
    elif action == 7:
        return grid.copy()
    else:
        return grid.copy()

def is_level_complete(grid):
    # Check if the grid contains the win state pattern
    # Based on the observed transitions, the win state is when the grid is filled with 5s
    # and the bottom row is filled with 0s
    if grid.shape != (64, 64):
        return False
    
    # Check if the bottom row is all 0s
    if not np.all(grid[63, :] == 0):
        return False
    
    # Check if the grid is filled with 5s (except the bottom row)
    if not np.all(grid[:63, :] == 5):
        return False
    
    return True