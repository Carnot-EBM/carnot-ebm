import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 2: Move Down
    if action == 2:
        # We observe a pattern of movement for blocks of colors 5 and 4.
        # The logic seems to be shifting these specific colored regions down by 3 rows.
        # Identify regions of color 5 and 4.
        mask_5 = (grid == 5)
        mask_4 = (grid == 4)
        
        # Create a copy to restore background
        new_grid = grid.copy()
        
        # Shift masks down by 3
        shifted_5 = np.roll(mask_5, 3, axis=0)
        shifted_4 = np.roll(mask_4, 3, axis=0)
        
        # Clear original positions of color 5 and 4
        # To avoid overwriting other important cells, we only clear those that are part of the same block.
        # Since we're moving them, they move from row r to r+3.
        # For each cell that is part of a block of 5s or 4s, set it back to background color 9.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[0]): # This should actually be col
                pass # Logic handled by mask shift
        
        # Re-apply shifted blocks
        # We need to be careful about boundaries. Roll wraps around.
        # The observed transitions show movement without wrap-around for these specific blocks.
        # a bit more complex: it's a<|channel>thought
        
        # Let's try a simpler approach based on the delta changes.
        # Action 2 shifts everything "down" effectively.
        # a bit more a thought: ACTION 2 seems to shift the entire state down.
        # But wait, look at the deltas. It moves blocks of 5s and 4s.
        # And it also updates some pixels at column 63 (color 11 -> 5).
        # And it modifies cells in columns 9-11 and 51-53.
        
        # Based on the provided data, we can see a very specific pattern:
        # Each ACTION 2 call shifts the 'active' block of colors 5 and 4 down by 3 rows.
        # Also, color 11 is changed to 5 at r=5, 6, 7... etc.
        # This looks like a cursor or progress indicator moving down the right edge.
        
        # Find current row of color 5 on the right edge (col 63)
        right_edge = grid[:, 63]
        idx_5 = np.where(right_edge == 5)[0]
        
        # The first action 2 happened after two action 3s.
        # Initial col 63 has 5x1 at r0,1,2. Then Action 3 changes r3c63 to 5.
        # Then Action 3 changes r4c63 to 5.
        # Then Action 2 changes r5c63 to 5.
        # So Action 2 increments the "progress" marker on the right edge.
        
        # Let's implement the shift logic for blocks 4 and 5.
        new_grid = grid.copy()
        
        # Shift blocks of 5 and 4 down by 3 rows.
        # We identify all cells that are NOT background (9), not borders (10, 11).
        # Specifically colors 4 and 5.
        for r in range(grid.shape[0]-1, -1, -1):
            for c in range(grid.shape[1]):
                if grid[r, c] == 5 or grid[r, c] == 4:
                    val = grid[r, c]
                    # Clear old position
                    new_grid[r, c] = 9
                    # Set new position
                    nr = r + 3
                    if nr < grid.shape[0]:
                        new_grid[nr, c] = val
        
        # Update progress marker on col 63
        current_marker_row = np.where(right_edge == 5)[0][-1] if len(idx_5) > 0 else -1
        new_grid[current_marker_row + 1, 63] = 5
        
        return new_grid

    # Action 3: Move Right/Left?
    if action == 3:
        # Observed ACTION 3 shifts blocks of 5s and 4s horizontally.
        # First call shifted them right (or left), second call shifted them again.
        # Looking at the deltas: r15c6 -> r15c3. This is a shift LEFT by 3 columns.
        
        new_grid = grid.copy()
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 5 or grid[r, c] == 4:
                    val = grid[r, c]
                    new_grid[r, c] = 9
                    nc = c - 3
                    if nc >= 0:
                        new_grid[r, nc] = val
        
        # Update progress marker on col 63
        right_edge = grid[:, 63]
        idx_5 = np.where(right_edge == 5)[0]
        current_marker_row = idx_5[-1] if len(idx_5) > 0 else -1
        new_grid[current_marker_row + 1, 63] = 5
        
        return new_grid

    return grid

def is_level_complete(grid):
    # The win state isn't explicitly provided as a full grid, but usually it involves
    # reaching a target area or completing the movement of blocks.
    # In this case, we don't have enough data to be sure, so we return False unless
    # some obvious condition is met.
    return False