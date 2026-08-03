import numpy as np

def engine(grid, action, data):
    """
    The game 'cn04' appears to be a puzzle where ACTION4 (right movement) 
    shifts a specific pattern of cells across the grid.
    Looking at the deltas for ACTION4:
    - It shifts columns by 3 units each time.
    - The changes occur in a complex vertical structure between rows 14 and 31.
    - There is also a small change at row 0 (a cell changing from 4 to 0).
    
    Based on the observed transitions, ACTION4 moves a "window" or "cursor" 
    of values to the right. Specifically, it seems to be shifting a set of 
    columns (width 3) that are being converted to color 10, while others 
    are converted back to 0 or other colors.
    """
    new_grid = grid.copy()
    
    if action == 4:
        # Action 4 shifts a window of width 3 to the right.
        # We need to identify which columns are currently modified and shift them.
        # In the provided trace, the shifted region starts at col 11, then 14, 17...
        # Let's find the current 'active' column index.
        # Row 0 has a marker: r0c16:0x1, then r0c17:0x1, etc.
        # This suggests the cursor position is tracked by the first 0 in row 0 starting from col 16.
        
        cursor_col = -1
        for c in range(16, 64):
            if grid[0, c] == 0:
                cursor_col = c
                break
        
        if cursor_col == -1: # Initial state before any ACTION4
            cursor_col = 15 # So that next is 16
            
        next_col = cursor_col + 1
        if next_col >= 64:
            return new_grid
            
        # Update the marker in row 0
        new_grid[0, next_col] = 0
        if cursor_col > 15:
            new_grid[0, cursor_col] = 4

        # The delta shows complex changes in rows 14-31.
        # It looks like columns [next_col+something : next_col+something+3] are set to 10,
        # and previous ones are reset.
        # Specifically, for r14-r16, cols [C, C+3) become 10, [C+3, C+6) become 0.
        # For r17-r19, cols [C, C+3) become 10, [C+3, C+6) become 0, [C+6, C+9) become 10...
        # This pattern repeats every 3 columns.
        
        # To simplify and match the observed deltas exactly:
        # We observe that ACTION4 shifts a "block" of modifications by 3 units.
        # Let's find where the current '10' blocks start in row 14.
        current_start = -1
        for c in range(11, 64):
            if grid[14, c] == 10:
                current_start = c
                break
        
        if current_start == -1:
            new_start = 11
        else:
            new_start = current_start + 3
            
        if new_start >= 64:
            return new_grid

        # Apply the shift to rows 14-31 based on the delta patterns
        # Row 14-16: col [new_start, new_start+3) -> 10, [new_start+3, new_start+6) -> 0
        for r in range(14, 17):
            new_grid[r, new_start : new_start+3] = 10
            new_grid[r, new_start+3 : new_start+6] = 0
            
        # Row 17-19: col [new_start, new_start+3) -> 10, [new_start+3, new_start+6) -> 0, [new_start+6, new_start+9) -> 10...
        for r in range(17, 20):
            new_grid[r, new_start : new_start+3] = 10
            new_grid[r, new_start+3 : new_start+6] = 0
            new_grid[r, new_start+6 : new_start+9] = 10
            new_grid[r, new_start+9 : new_start+12] = 0

        # Row 20-22: similar but shifted
        for r in range(20, 23):
            new_grid[r, new_start : new_start+3] = 10
            new_grid[r, new_start+3 : new_start+6] = 0 # This is a simplification
            new_grid[r, new_start+6 : new_start+9] = 10
            new_grid[r, new_start+9 : new_start+12] = 0
            
        # Rows 23-31 follow the pattern of shifting blocks of 3.
        for r in range(23, 32):
            new_grid[r, new_start : new_start+3] = 10
            new_grid[r, new_start+3 : new_start+6] = 0
            new_grid[r, new_start+6 : new_start+9] = 10
            new_grid[r, new_start+9 : new_start+12] = 0

    return new_grid

def is_level_complete(grid):
    # No win state provided in observations, assume False unless specific condition met.
    # Usually ARC levels are complete when a target pattern is reached.
    return False