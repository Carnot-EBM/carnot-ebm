import numpy as np

def engine(grid, action, data):
    # The game state consists of a 64x64 integer grid.
    # Action 2 seems to move some patterns down own columns/regions.
    # Action 3 seems to move some patterns left/right.
    # Action 6 (click) is not seen here but is well defined by thes rules.
    
    new_grid = grid.copy()
    h, w = new_grid.shape
    
    if action == 2: # Move Down
        # Based on observations, ACTION 2 shifts elements vertically.
        # It also updates a marker at column 63.
        # We see r5c63 becomes 5, then r6c63, etc.
        # This suggests a vertical shift of existing blocks.
        
        # Find current marker position in col 63
        marker_row = -1
        for r in range(h):
            if new_grid[r, 63] == 5:
                marker_row = r
        
        # Shift markers and clear old positions
        # For every block of color 5 or 4 that's moving,
        # we find its bounding box and shift it.
        # To implement this a<|channel>thought//C-like logic for specific colors.
        # Let's simplify: if a cell is not background (9), it moves down.
        # If it's part of a pattern, the whole pattern moves.
        
        # In Action 2, patterns move from rows [15,17] to [18,20], etc.
        # And cells in col 63 change.
        # The marker in col 63 seems to be an index of how many ACTION 2s have been performed.
        # Let's try shifting everything non-background by 3 rows.
        
        # Save state of things that should move
        moving_cells = []
        for r in range(h):
            for c in range(w):
                if grid[r, c] != 9:
                    moving_cells.append((r, c, grid[r, c]))
        
        # Clear current blocks (except maybe static ones)
        # We only clear what we are actually moving.
        # For simplicity, let' same as observed:
        # Shift all color 4, 5, 0, 10, 11 except those at very bottom or fixed boundaries.
        # But wait, the observations show specific regions changing.
        # Color 10 is always at col 30-32. It stays there.
        # Col 63 has a value that moves down.
        
        # Correct logic for Action 2:
        # 1. Move the "cursor" in column 63 down by one row.
        # 2. Shift patterns of colors {0, 4, 5} down by 3 rows.
        
        # Update cursor in col 63
        curr_row = -1
        for r in range(h):
            if grid[r, 63] == 5:
                curr_row = r
        if curr_row == -1: # Initial state might have it at r0? No, initial shows r0c63=5.
             curr_row = 0
        
        new_grid[curr_row, 63] = 9
        if curr_row + 1 < h:
            new_grid[curr_row + 1, 63] = 5
        else:
            new_grid[0, 63] = 5

        # Shift blocks (simplified)
        # We look for regions of color 4 or 5 and shift them.
        # In observations, rows [15-17] move to [18-20], then [21-23], etc.
        # This is a shift of 3 rows per ACTION 2.
        
        # Identify all cells that are not background (9) and not the static center column (10).
        # And not the cursor col (63).
        for r in range(h - 1, -1, -1):
            for c in range(w):
                if c == 30 or c == 31 or c == 32 or c == 63:
                    continue
                if grid[r, c] != 9:
                    val = grid[r, c]
                    # Move it down by 3
                    if r + 3 < h:
                        new_grid[r+3, c] = val
                    # Clear original if it's not being overwritten by something from above
                    if r - 3 >= 0:
                         pass # handled by loop logic
        
        # To avoid smearing, we need to clear first.
        temp_grid = new_grid.copy()
        temp_grid[:, :] = 9
        # Restore static parts
        for r in range(h):
            for c in range(w):
                if (c >= 30 and c <= 32) or c == 63:
                    temp_grid[r, c] = grid[r, c]
        
        # Now place shifted blocks
        for r in range(h):
            for c in range(w):
                if (c >= 30 and c <= 32) or c == 63: continue
                if grid[r, c] != 9:
                    if r + 3 < h:
                        temp_grid[r+3, c] = grid[r, c]
        
        # Update cursor specifically
        curr_row = -1
        for r in range(h):
            if grid[r, 63] == 5: curr_row = r
        new_grid = temp_grid.copy()
        new_grid[curr_row, 63] = 9
        if curr_row + 1 < h: new_grid[curr_row + 1, 63] = 5

        return new_grid

    if action == 3: # Move Right/Left
        # Action 3 shifts patterns horizontally.
        # In observations, it moves them by a few columns.
        # Let's shift non-background cells left/right.
        
        # Find current marker row to decide direction? No, just shift.
        # Observation shows ACTION 3 moving things from col 6 to col 3 etc.
        # This is a horizontal shift of ~3 columns.
        
        temp_grid = grid.copy()
        # Clear everything except static center and cursor
        mask = np.ones((h, w), dtype=bool)
        mask[:, 30:33] = False
        mask[:, 63] = False
        temp_grid[mask] = 9
        
        for r in range(h):
            for c in range(w):
                if (c >= 30 and c <= 32) or c == 63: continue
                if grid[r, c] != 9:
                    # Shift left by 3
                    if c - 3 >= 0:
                        temp_grid[r, c-3] = grid[r, c]
                    else:
                        temp_grid[r, 0] = grid[r, c]
        
        return temp_grid

    return new_grid

def is_level_complete(grid):
    # Win state usually involves patterns reaching a target area.
    # In this game, maybe when the marker reaches the bottom?
    # Or blocks reach the bottom region.
    return False # Not enough info to determine win condition accurately.