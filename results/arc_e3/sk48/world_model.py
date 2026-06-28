import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 1:
        if data is None:
            return grid
        # Action 1 is a vertical click at (x, y)
        x, y = data['x'], data['y']
        h, w = grid.shape
        # Create a mask for the clicked column
        col_mask = np.zeros((h, w), dtype=int)
        col_mask[:, x] = 1
        
        # Apply the click effect: toggle cells in the column
        # Based on observed transitions, the click toggles cells in a specific pattern
        # The pattern seems to be a vertical line of toggles
        # We'll implement a simple toggle mechanism
        
        # Convert grid to list of lists for easier manipulation
        grid_list = grid.tolist()
        
        # Toggle cells in the column
        for i in range(h):
            if col_mask[i, x] == 1:
                grid_list[i][x] = 1 - grid_list[i][x]
        
        return np.array(grid_list)
    elif action == 2:
        if data is None:
            return grid
        # Action 2 is a horizontal click at (x, y)
        x, y = data['x'], data['y']
        h, w = grid.shape
        # Create a mask for the clicked row
        row_mask = np.zeros((h, w), dtype=int)
        row_mask[y, :] = 1
        
        # Apply the click effect: toggle cells in the row
        grid_list = grid.tolist()
        
        # Toggle cells in the row
        for i in range(w):
            if row_mask[y, i] == 1:
                grid_list[y][i] = 1 - grid_list[y][i]
        
        return np.array(grid_list)
    else:
        return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on observed transitions, a win state is when the grid is fully filled
    # or when a specific pattern is achieved
    # For simplicity, we'll check if the grid is fully filled with non-zero values
    return np.all(grid != 0)

import numpy as np

def is_level_complete(grid):
    # Flatten grid to 1D
    flat = grid.flatten()
    # Check if all elements are equal to the most frequent element (win state)
    if len(flat) == 0:
        return False
    # Use numpy to find unique values and their counts
    unique, counts = np.unique(flat, return_counts=True)
    # If the most frequent element appears more than half the time, it's a win
    return counts[-1] > len(flat) / 2
