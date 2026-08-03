import numpy as np

def engine(grid, action, data):
    """
    The observed transitions show ACTION4 (which usually corresponds to a 
    directional key like 'Right' in many ARC-AGI environments) moving a complex 
    pattern of cells across the grid.
    
    Looking at the deltas:
    - The pattern consists of several blocks of color 10 and some gaps/other colors.
    - Each ACTION4 shifts this entire structure to the right by exactly 3 columns.
    - There is also a single cell changing at r0c16, r0c17... which suggests a cursor or marker.
    
    Rule Induction:
    ACTION4 moves all non-background (color 10, 0, 8, 14) elements that are part of the 
    central "object" 3 units to the right. However, looking closer at the delta, it seems 
    to be shifting specific regions. Specifically, it looks like a translation of the 
    entire active area of the board.
    """
    new_grid = grid.copy()
    
    if action == 4:
        # Shift everything to the right by 3 pixels.
        # Based on the deltas, the shift affects rows from 0 down to 31.
        # We can implement this as a general shift for the whole grid, 
        # but we must handle boundaries.
        shift = 3
        # To match the observed behavior where cells 'disappear' or change at the edges:
        # The pattern is shifted and wrapped or clipped.
        # Let's use np.roll for simplicity if it fits, or manual slice.
        
        # Observed: r0c16 becomes 0, then r0c17... This is a marker moving.
        # The main body shifts.
        
        # Create a temporary grid to hold the shifted values
        temp_grid = np.full_like(grid, 10) # Background color is 10
        
        # In ARC-AGI games of this type, usually the "background" (color 10 here) 
        # stays put and only the "foreground" moves.
        # However, the delta shows that color 10 replaces other colors too.
        # It looks like a translation of the entire image content relative to background 10.
        
        # Shift logic:
        # For each row, move elements from col j to j+3.
        for r in range(grid.shape[0]):
            row = grid[r]
            new_row = np.full(grid.shape[1], 10)
            for c in range(grid.shape[1] - shift):
                new_row[c + shift] = row[c]
            # Handle the left edge filling with background
            for c in range(shift):
                new_row[c] = 10
            new_grid[r] = new_row
            
    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided, but typically it involves 
    reaching a specific configuration or clearing an area.
    Without a WIN STATE example, we return False unless a known condition is met.
    """
    # In most ARC tasks, completion is defined by the final target pattern.
    # Since no target was given, we assume the level continues until external stop.
    return False