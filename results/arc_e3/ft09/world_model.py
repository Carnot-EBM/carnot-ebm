import numpy as np

def engine(grid, action, data):
    """
    Simulates the next state of the grid based on the action.
    The game appears to be a simple click-and-change-color mechanic.
    Clicking a specific location (x, y) triggers a change in a 6x7 block
    of cells in the grid (rows 52-57, cols 36-41 or similar).
    The color changes from 8 to 9.
    """
    grid = grid.copy()
    H, W = grid.shape
    
    if action == 6:
        # Action 6 is a click.
        px, py = data['x'], data['y']
        
        # The observed changes happen in a specific region:
        # Rows 52-57, Columns 36-41 (approx).
        # The click seems to target a specific column based on x.
        # Based on observations:
        # x=46 -> col 44
        # x=38 -> col 36
        # x=54 -> col 52
        # It seems col = x - 2.
        
        if px < 0 or px >= W or py < 0 or py >= H:
            return grid
            
        target_col = px - 2
        
        # Check if the target column is within the valid range for the observed pattern
        # The pattern affects rows 52-57.
        if target_col < 36 or target_col >= 42:
            return grid
            
        # The observed change is a 6x6 block (or 6x7) changing from 8 to 9.
        # Rows 52-57, Cols target_col to target_col+5.
        # Also, row 63 changes at col target_col+6 to 11.
        
        # Apply the change to the grid
        # Rows 52-57, Cols target_col to target_col+5
        grid[52:58, target_col:target_col+6] = 9
        
        # Row 63, Col target_col+6
        grid[63, target_col+6] = 11
        
        return grid

    return grid

def is_level_complete(grid):
    """
    Checks if the level is complete.
    Based on the observations, the level seems to be complete when
    the specific pattern of changes has been made.
    However, without a clear win condition in the data, we return False.
    """
    return False