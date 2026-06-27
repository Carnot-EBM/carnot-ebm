import numpy as np

def engine(grid, action, data):
    grid = grid.copy()
    H, W = grid.shape
    
    if action == 1:
        # Action 1: Fill a vertical column with a specific pattern
        # Based on observed transitions, Action 1 fills a column at a specific position
        # The pattern seems to be: 0s, then 14s, then 1s, then 1s
        # The column position is determined by the data
        if data is not None:
            x, y = data['x'], data['y']
            col = x // 1
            row_start = y // 1
            # Fill the column with the pattern
            # Pattern: 0s, then 14s, then 1s
            grid[row_start:row_start+4, col] = 0
            grid[row_start+4:row_start+8, col] = 14
            grid[row_start+8:row_start+12, col] = 1
        else:
            # Default pattern if no data
            grid[:, 32] = 0
            grid[4:8, 32] = 14
            grid[8:12, 32] = 1
            
    elif action == 2:
        # Action 2: Similar to Action 1 but with different pattern
        # Pattern: 1s, then 14s, then 1s, then 0s
        if data is not None:
            x, y = data['x'], data['y']
            col = x // 1
            row_start = y // 1
            grid[row_start:row_start+4, col] = 1
            grid[row_start+4:row_start+8, col] = 14
            grid[row_start+8:row_start+12, col] = 1
            grid[row_start+12:row_start+16, col] = 0
        else:
            # Default pattern
            grid[:, 32] = 1
            grid[4:8, 32] = 14
            grid[8:12, 32] = 1
            grid[12:16, 32] = 0
            
    elif action == 4:
        # Action 4: Fill a rectangular region with a pattern
        # Pattern: 1s, then 14s, then 0s
        if data is not None:
            x, y = data['x'], data['y']
            col = x // 1
            row_start = y // 1
            # Fill a 4x8 region
            grid[row_start:row_start+4, col:col+8] = 1
            grid[row_start+4:row_start+8, col:col+8] = 14
            grid[row_start+8:row_start+12, col:col+8] = 0
        else:
            # Default pattern
            grid[52:56, 32:40] = 1
            grid[56:60, 32:40] = 14
            grid[60:64, 32:40] = 0
            
    elif action == 5:
        # Action 5: No change
        pass
        
    elif action == 6:
        # Action 6: Click action
        if data is not None:
            x, y = data['x'], data['y']
            grid[y, x] = 4
            
    elif action == 7:
        # Action 7: No change
        pass
        
    return grid

def is_level_complete(grid):
    # Check if the grid is in a win state
    # Based on the observed transitions, a win state is when the grid is filled with a specific pattern
    # The pattern seems to be: 1s, then 14s, then 0s
    H, W = grid.shape
    
    # Check if the grid is filled with the win pattern
    win_pattern = np.zeros((H, W), dtype=int)
    win_pattern[0:H//4, :] = 1
    win_pattern[H//4:3*H//4, :] = 14
    win_pattern[3*H//4:H, :] = 0
    
    return np.array_equal(grid, win_pattern)