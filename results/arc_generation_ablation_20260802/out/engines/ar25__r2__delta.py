import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the action taken.
    """
    out = grid.copy()
    h, w = out.shape
    
    if action == 2: # Move Down / ACTION2
        # The observed transitions show a pattern where blocks move down.
        # Specifically, there are two vertical columns of 'active' areas (cols 3-11 and 51-59).
        # Action 2 seems to shift these patterns downward by 3 rows.
        # We need to find the current positions of the blocks and shift them.
        
        # Find all non-background (color 9) cells that aren't static walls/markers.
        # Color 10 is a central wall, color 11 is right edge marker.
        # In the same way we<|channel>thought about the same coordinates, let's look at the observations.
        # Let's identify the "blocks" in the active zones.
        # Zone A: cols 3-11; Zone B: cols 51-59.
        # Shift everything in those zones down by 3? No, it looks like they slide.
        
        # Based on the delta evidence:
        # Transition 3: r15c3:9x9, r15c51:9x9... shifted from previous state.
        # It moves a block of size 3x9 down by 3 rows each time ACTION2 is called.
        # Also updates a marker on the far right column (col 63).
        
        # Identify the 'block' currently moving.
        # The block consists of colors other than 9.
        # We need to find the top-most row containing a cell with value != 9 in the active zones.
        # Find current position of the blocks.
        # Look for cells that are not background (color 9) and not static walls (color 10 or 11).
        # In the active zones (cols 3-11 and 51-59), look for values {4, 5, 0}.
        # Let's assume the movement is a simple shift of all non-background pixels in these zones.
        
        # Correct approach based on observations:
        # Action 2 shifts everything in cols 3-11 and 51-59 down by 3 rows.
        # And it increments a marker at col 63.
        
        # Save the content of the columns before shifting.
        # For the two active zones:
        # Zone A: [:, 3:12]
        # Zone B: [:, 51:60]
        # Shift them down by 3.
        
        # To implement this simply:
        # The evidence shows the block moves from r15->r18->r21...
        # It also seems to "clear" the previous area.
        
        # Find current top row of the moving blocks in active zones.
        # mask = (grid[:, 3:12] != 9) | (grid[:, 51:60] != 9)
        # But wait, there are static elements too? No, they seem to move together.
        
        # Let's just perform a vertical shift of the entire column slices.
        # We need to handle boundaries.
        # The observed deltas show that cells become color 9 when the block leaves.
        # And new cells get values when the block arrives.
        
        # Specifically for ACTION2:
        # Marker at col 63 increments its row index by 1 each time.
        # Block shifts down by 3 rows.
        
        # Let's find where the marker is currently.
        marker_row = -1
        for r in range(h):
            if out[r, 63] == 5:
                marker_row = r
                break
        
        # Move marker: clear old, set new.
        if marker_row != -1 and marker_row + 1 < h:
            out[marker_row, 63] = 9 # Or whatever background was
            out[marker_row + 1, 63] = 5
        elif marker_row == -1:
            # Initial state might not have it or it starts at r0?
            # Looking at INITIAL GRID, r0c63 is 5x1. So start at r0.
            out[0, 63] = 5
            
        # Shift blocks in active zones.
        # We only shift cells that are part of the "moving block" (colors {4, 5, 0}).
        # The moving block seems to be a specific pattern.
        # Instead of complex logic, let's just shift all non-background pixels in those columns down by 3.
        
        zones = [(3, 12), (51, 60)]
        for start_col, end_col in zones:
            # Store current column values
            col_data = grid[:, start_col:end_col].copy()
            # Clear the zone in output
            out[:, start_col:end_col] = 9
            # Shift and place back
            for r in range(h - 3):
                if np.any(col_data[r] != 9):
                    out[r + 3, start_col:end_col] = col_data[r]

    elif action == 3: # Move Right / ACTION3
        # Based on observations, Action 3 shifts the patterns horizontally.
        # Looking at deltas: r15c6->r15c3... wait, it shifted left?
        # Let's check: "changed cells ... r15c6:5x3 ... then r15c3:5x3".
        # That is a move to the LEFT.
        # But let's look at the marker: r3c63 -> r4c63. The marker still moves down.
        # Wait, looking at the first two ACTION3 calls:
        # Call 1: r15c6:5x3, r15c15:9x3...
        # Call 2: r15c3:5x3, r15c12:9x3...
        # It moved from c6 to c3. That is -3 columns.
        
        # Marker movement in ACTION3:
        marker_row = -1
        for r in range(h):
            if grid[r, 63] == 5:
                marker_row = r
                break
        if marker_row != -1 and marker_row + 1 < h:
            out[marker_row, 63] = 9
            out[marker_row + 1, 63] = 5

        # Shift blocks horizontally by -3 (Left).
        zones_rows = [15, 16, 17, 18, 19, 20, 21, 22, 23] # Rows involved in Action 3 deltas
        for r in zones_rows:
            # We shift the content of the active zones left by 3.
            # Zone A: cols 3-12; Zone B: cols 51-60.
            # This is tricky because they are separate zones.
            # Let's just shift the whole row for those specific rows?
            # No, only within the active zones.
            
            # For simplicity, let's shift all non-background pixels in these rows left by 3.
            row_data = grid[r, :].copy()
            out[r, :] = grid[r, :] # reset to original
            # Only change cells that were part of the block
            # In ACTION3, it seems to move a block from c6->c3 and c45->c42 etc.
            # Let's try shifting everything in the row left by 3, but only if it's not background.
            # Actually, looking at the delta: "r15c6:5x3" becomes "r15c3:5x3".
            # It's a simple translation.
            
            # To avoid destroying walls (col 30-33), we only shift within the two zones.
            for start_col, end_col in [(3, 12), (51, 60)]:
                zone_data = grid[r, start_col:end_col].copy()
                out[r, start_col:end_col] = 9
                for c in range(len(zone_data)):
                    if zone_data[c] != 9:
                        new_c = c - 3
                        if new_c >= 0:
                            out[r, start_col + new_c] = zone_data[c]

    return out

def is_level_complete(grid):
    """
    Returns True if the win state is reached.
    Based on typical ARC games, this might be when blocks reach a target or are aligned.
    Without a WIN STATE example, we assume it's complete when the marker reaches the bottom.
    """
    h, w = grid.shape
    # Check if any block has reached the bottom active area (rows 45-53)
    # Or if the marker at col 63 has reached the bottom.
    if grid[h-1, 63] == 5:
        return True
    
    # Also check for specific patterns in the bottom rows.
    # In INITIAL GRID, r45-r53 have color 11 blocks.
    # If those are replaced by colors {4, 5, 0}, it might be a win.
    bottom_zone = grid[45:54, 3:12]
    if np.any((bottom_zone != 9) & (bottom_zone != 11)):
        return True
        
    return False