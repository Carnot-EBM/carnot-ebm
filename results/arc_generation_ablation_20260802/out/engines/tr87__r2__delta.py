import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the action.
    """
    out = grid.copy()
    
    if action == 4: # ACTION4 (Move Right/Down?) - Based on observed deltas, it shifts some specific blocks.
        # This looks like it's moving a set of '0' values (blocks) across the board.
        # We need to find where these '0's are currently located.
        # Find all indices of color 0.
        zeros = np.argwhere(grid == 0)
        if len(zeros) > 0:
            # The patterns of zeros shift by 7 columns each time ACTION4 is triggered.
            # The pattern is r48c15-20, r49c15, r49c19, r59c15, r59c19, r60c15-20 and r48c22-27, etc.
            # Let's identify the current "block" of zeros.
            # Look for the first zero at row 48.
            first_zero_col = -1
            for c in range(64):
                if grid[48, c] == 0:
                    first_zero_col = c
                    break
            
            if first_zero_col != -1:
                # Shift everything that was 0 to be 3 and everything that became 0// a new set of 0s shifted right by 7.
                # Current positions of the cells that will become non-zero (the old zeros).
                old_zeros = np.argwhere(grid == 0)
                
                # New positions of the zeros own their shape.
        
        # Since we are inducing from very limited data, let' same as observed deltas.
        # The ACTION4 shifts a specific pattern of zeros.
        #<|channel>thought: ACTION4 is 'Right'.
        # If action is 4, shift the blocks of color 0 by 7 columns to the right.
        # Find all coordinates of color 0.
        coords_0 = np.argwhere(grid == 0)
        if len(coords_0) > 0:
            # Save current state to the block of 0s.
            # mask = (grid == 0)
            # For each coordinate in row 48, 49, 59, 60.
            # {r, c} -> {r, c+7}
            # We need to know what replaces the old 0s.
            # Initial grid has 0s at r48c15-20 and r60c15-20.
            # Transition 1: ACTION4 changes r48c15:3x5, r48c22:0x5, etc.
            # This means cells that were 0 become 3, and cells that were something else become 0.
            # The values being shifted are the "holes" or "blocks".
            
            # Let's implement a simple shift for the zeros pattern.
            # Shift everything that is 0 by +7 columns.
            # Also handle the specific rows mentioned in deltas.
            rows_to_shift = [48, 49, 59, 60]
            for r in rows_to_shift:
                # Find all cols where grid[r, c] == 0
                cols = np.where(grid[r, :] == 0)[0]
                for c in cols:
                    out[r, c] = 3 # Replace old zero with background color 3
                    if c + 7 < 64:
                        out[r, c+7] = 0
    
    elif action == 2: # ACTION2 (Move Down?) - Based on observed deltas, it shifts some blocks of colors 5/7.
        # ACTION2 seems to move things in the region r52-56.
        # Find coordinates of color 5 and 7.
        coords_5 = np.argwhere(grid == 5)
        coords_7 = np.argwhere(grid == 7)
        # The delta shows changes in r52c22, r53c22, etc.
        # This looks like a shift or toggle.
        # Let's try to implement a simple movement for these specific cells.
        # In Transition 2, ACTION2 changed r52c22:5x1, r52c26:5x1...
        # It moves things around in that block.
        pass

    elif action == 1: # ACTION1 (Move Left?) - Based on observed deltas, it shifts some blocks of colors 5/7.
        # Similar to ACTION2, but different columns.
        pass

    return out

def is_level_complete(grid):
    """
    Check if the level is complete.
    """
    # No win state provided, assume not complete unless specified.
    # return True if grid[63, 63] == something?
    # Looking at the transitions, r63c62:4x1, r63c61:4x1, r63c60:4x1...
    # This suggests color 4 is moving leftwards along row 63.
    # Maybe completion is when color 4 reaches column 0?
    return False