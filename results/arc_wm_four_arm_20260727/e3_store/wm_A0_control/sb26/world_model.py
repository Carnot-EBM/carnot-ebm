import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        # Convert pixel to logical
        r, c = py // 1, px // 1
        # Check if the click is on a specific target
        # Based on observed transitions, clicking at (19, 58) or (35, 58) or (40, 29) triggers changes
        # The changes involve placing 0s and 12s in specific patterns
        # This is a complex pattern matching based on the observed data
        
        # Simplified rule: if the click is in the bottom area (rows 56-61), it places 0s
        # if the click is in the middle area (rows 27-32), it places 12s and 14s
        
        # Check if the click is in the bottom area
        if 56 <= r <= 61:
            # Place 0s in a vertical line at column c
            # The observed data shows 0s placed at columns 17, 22, 33, 38
            # This suggests a pattern based on the click position
            # For simplicity, we place 0s in a vertical line at the clicked column
            # and also place 12s in the rows above
            
            # Place 0s in the clicked row and adjacent rows
            for i in range(r, r + 6):
                if i < 64:
                    grid[i, c] = 0
            # Place 12s in the rows above
            for i in range(r - 6, r):
                if i >= 0:
                    grid[i, c] = 12
        elif 27 <= r <= 32:
            # Place 12s and 14s in a vertical line at column c
            for i in range(r, r + 6):
                if i < 64:
                    grid[i, c] = 12
            # Place 14s in the rows above
            for i in range(r - 6, r):
                if i >= 0:
                    grid[i, c] = 14
        else:
            # No change
            pass
    elif action == 7:
        # No change
        pass
    else:
        # No change
        pass
    return grid

def is_level_complete(grid):
    # Check if the grid is complete
    # Based on the observed data, the level is complete when certain patterns are formed
    # This is a simplified check
    # Check if the grid has the expected patterns
    # For simplicity, we check if the grid has the expected number of 0s and 12s
    # This is a heuristic check
    
    # Count the number of 0s and 12s
    num_zeros = np.sum(grid == 0)
    num_twelves = np.sum(grid == 12)
    
    # Check if the grid is complete
    # This is a simplified check
    return num_zeros > 0 and num_twelves > 0