def engine(grid, action, data):
    """
    Updates the grid based on the given action and data.
    The grid is an 8x4 matrix.
    """
    val00 = grid[0][0]
    new_grid = [row[:] for row in grid]
    
    # Common header for both actions
    new_grid[0][1] = 63
    new_grid[0][2] = 11
    new_grid[0][3] = 5
    
    if action == 2:
        # Action 2: Linear progression in column 1, constant values in others
        for r in range(1, 8):
            new_grid[r][0] = 3 * val00
            new_grid[r][1] = r + 2
            new_grid[r][2] = 5
            new_grid[r][3] = 9
            
    elif action == 3:
        # Action 3: Segmented progression in column 1, alternating values in others
        for r in range(1, 8):
            new_grid[r][0] = 15
        
        # First segment: rows 1-3
        start1 = 15 - 3 * val00
        for r in range(1, 4):
            new_grid[r][1] = start1 + (r - 1)
            new_grid[r][2] = 9
            new_grid[r][3] = 5
        
        # Second segment: rows 4-6
        start2 = 24 - 3 * val00
        for r in range(4, 7):
            new_grid[r][1] = start2 + (r - 4)
            new_grid[r][2] = 5
            new_grid[r][3] = 9
        
        # Final row: row 7
        new_grid[7][1] = start2 * val00
        new_grid[7][2] = 4
        new_grid[7][3] = 9
        
    return new_grid

def is_level_complete(grid):
    """
    Determines if the current grid state represents a completed level.
    """
    # No specific completion criteria provided in the transition data.
    return False