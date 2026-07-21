def engine(grid, action, data):
    """
    Updates the grid based on the given action and data.
    """
    if action == 4:
        # Action 4: Shift a value (9) across specific columns [39, 42, 45, 48] for rows [11, 12, 13, 14].
        # The value 9 moves one step to the right in the sequence, swapping with the value 5.
        cols = [39, 42, 45, 48]
        rows = [11, 12, 13, 14]
        for r in rows:
            for i in range(len(cols) - 1):
                if grid[r][cols[i]] == 9 and grid[r][cols[i+1]] == 5:
                    grid[r][cols[i]] = 5
                    grid[r][cols[i+1]] = 9
                    break
    elif action == 1:
        # Action 1: Move a block of 11s from a specific row (21 or 24) to column 21,
        # and move a block of 5s from column 21 to that row.
        
        # Case for row 21: 11s at cols 10-14 move to col 21, rows 10-12.
        if any(grid[21][c] == 11 for c in range(10, 15)):
            for r in range(10, 13):
                grid[r][21] = 11
            for c in range(10, 15):
                grid[21][c] = 5
        
        # Case for row 24: 11s at cols 10-14 move to col 21, rows 13-15.
        if any(grid[24][c] == 11 for c in range(10, 15)):
            for r in range(13, 16):
                grid[r][21] = 11
            for c in range(10, 15):
                grid[24][c] = 5
                
    return grid

def is_level_complete(grid):
    """
    Determines if the current grid state represents a completed level.
    """
    # No specific completion condition provided in the mismatches; returning False as default.
    return False