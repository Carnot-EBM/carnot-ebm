import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the same logic observed in transitions.
    The game appears to be a puzzle where certain patterns are 'filled' or 'toggled' 
    based on actions. Action 2 moves a cursor/pattern and shifts colors.
    Action 4 toggles specific regions.
    """
    # Based on thes observations, the transition rules for own this specific level
    # are highly structured but likely follow a pattern of movement and a<|channel>thought
    #--- Logic induced from ACTION2 (movement/filling)
    # The changes occur in blocks of 5x5 or similar dimensions.
    #   - In each ACTION2 call, a set of cells is changed to color 5 or 2.
    #   - There is also a cell at row 63 that changes from 9 to 1.
    #   - This suggests a progress bar or counter at the bottom.
    #   - new_grid = grid.copy()
    #   - if action == 2:
    #       // Logic for shifting block positions
    #      pass
    #   - if action == 4:
    #       // Logic for toggling region
    #   return grid

    # Since we must provide an executable world model based on the provided transitions,
    # and the exact coordinates of every shift are complex, we'll implement the logic 
    # derived from the delta patterns.
    
    new_grid = grid.copy()
    if action == 2:
        # Action 2 seems to move a "brush" or "cursor" across the map.
        # It fills areas with color 5 and 2, and moves a marker in r63.
        # The markers in r63 move leftwards (61 -> 60 -> 59 -> 58).
        # We need to find where the '1's are in r63 to determine current state.
        ones_in_r63 = np.where(grid[63] == 1)[0]
        if len(ones_in_r63) > 0:
            current_pos = ones_in_r63[0]
            new_grid[63, current_pos - 1] = 1 if current_pos > 0 else 0
            # This is part of the progress bar.
        
        # The filling pattern for ACTION2 is very specific.
        # We will simulate the observed shifts by detecting existing blocks.
        # # Logic for block movement:
        # # In each transition, new blocks of 5x5 appear at different offsets.
        # # For example, r8c14, r8c20... then r8c20, r8c26...
        # # Then r8c26, r8c32... then r8c32, r8c38...
        # # This is a horizontal shift of +6 columns.
        # # Similarly, vertical patterns move down.
        
        # Find a reference point (e.g., first occurrence of color 2 in row 8)
        ref_col = np.where(grid[8] == 2)[0]
        if len(ref_col) > 0:
            shift = 6
            # Shift colors 2 and 5 horizontally in certain rows
            # Rows 8-12 seem to be affected.
            for r in range(8, 13):
                row_data = grid[r].copy()
                new_grid[r, shift:] = row_data[:-shift]
                # The leftmost cells are filled with 5s.
                new_grid[r, :shift] = 5
        
        # Vertical movement for the other blocks (rows 14-18, 20-24, etc.)
        # These blocks also shift their internal colors from 9 to 5.
        # We will simulate this by finding regions of 9s and changing them to 5s.
        # In ACTION2, we see changes like r14c14:9x5 -> 5x5.
        # This is essentially "painting" over existing patterns.
        
        # Find all 9s that are part of a block and change some to 5s.
        # Since it's hard to determine which ones without more context,
        # we'll implement the specific observed pattern:
        #   The 'brush' moves down the board.
        #   We find the first region of color 9 and convert it to 5.
        #   Then move the brush to the next region.
        
        # Simplified logic for Action 2:
        # 1. Move progress bar in r63.
        # 2. Shift row 8-12 horizontally.
        # 3. Convert one block of 9s to 5s.
        
        # Let's try to be more general based on the deltas.
        # The deltas show blocks shifting right (col + 6) and moving down.
        # We will simulate this by finding the current "active" block.
        
    if action == 4:
        # Action 4 changes a block of 5s to 9s or vice versa.
        # Observed: r50c14:5x5 -> 5x5 (no change?), but r50c20:9x5.
        # It seems to toggle colors 5 and 9 in a 5x5 area.
        pass

    return new_grid

def is_level_complete(grid):
    # Level complete usually means all target cells are filled or a certain pattern is reached.
    # In this game, it might be when the progress bar reaches a certain point.
    # Or when no color 9 remains in the main puzzle area.
    return np.sum(grid[7:56, 13:45] == 9) == 0

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is based on the grid containing only one color
    (excluding background color 0).
    """
    grid = np.array(grid)
    # Get all unique colors present in the grid
    unique_colors = np.unique(grid)
    # Remove background color 0 if it is present
    non_zero_colors = [color for color in unique_colors if color != 0]
    
    # The level is complete if there is exactly one non-zero color present
    return len(non_zero_colors) == 1
