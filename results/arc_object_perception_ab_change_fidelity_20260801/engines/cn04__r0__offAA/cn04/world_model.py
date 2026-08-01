import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action 4 is a rightward shift of certain elements in the same row/column structure.
    # Based on observed transitions, ACTION4 shifts a pattern of cells to the right by 1 column.
    # We identify the "active" region that moves.
    # The active region seems to be defined by columns 16-47 and rows 14-31.
    # The active region consists of blocks of color 0 or other colors.
    #
    # Let's refine the logic based on the<|channel>thought process:
    # It looks like a puzzle where a vertical strip of content is moving right.
    #
    # Specifically, looking at the deltas:
    # r0c16:0x1 -> r0c17:0x1 -> r0c18:0x1 ...
    # This indicates a single cell at (0, 16) is moving right.
    #
    # For the rest of the board, it's more complex.
    # In each transition, the cells shifted are those that were not color 10.
    # Color 10 is the background.
    #
    # Let's implement a simple shift for all non-background pixels in specific ranges.
    # H, W = grid.shape
    # mask = (grid != 10)
    # next_grid = grid.copy()
    #
    # However, the observed transitions show very specific columns being changed to 10 and others changing from 10.
    # This is exactly what happens when you shift a pattern.
    #
    # The active region seems to be everything except the far left/right borders.
    #
    # Let's try shifting all non-10 pixels one step to the right if they are within bounds.
    
    if action == 4:
        next_grid = grid.copy()
        # We need to identify which pixels move. 
        # Based on the deltas, only certain rows and columns are affected.
        # Row 0 moves. Rows 14-31 move.
        # Column range is roughly 16 to 47.
        
        # To avoid destroying the rest of the board, we only shift cells that are not color 10
        # AND are part of the moving structure.
        # In this game, it seems any cell that isn't background (10) in those rows shifts.
        
        rows_to_shift = [0] + list(range(14, 32))
        for r in rows_to_shift:
            # Shift row r by 1 column to the right
            # But only for the "active" content.
            # Find first and last non-10 index in the row
            non_10 = np.where(grid[r] != 10)[0]
            if len(non_10) > 0:
                first, last = non_10[0], non_10[-1]
                # The observed transitions show a very specific window shifting.
                # We shift everything from 'first' to 'last' one step right.
                # This means grid[r, first] becomes 10, and grid[r, last+1] becomes what was at grid[r, last].
                
                # To implement this correctly as a block shift:
                row_segment = grid[r, first : last + 1]
                next_grid[r, first] = 10
                next_grid[r, first + 1 : last + 2 if last + 2 <= W else W] = \
                    np.pad(row_segment, (1, 0))[:-1] # This is not quite right.
        
        # Let's use a simpler approach: for each cell that isn't background, move it.
        # But we must do it in reverse order to avoid overwriting.
        return next_grid

    return grid

def engine(grid, action, data):
    if action == 4:
        new_grid = grid.copy()
        H, W = grid.shape
        rows_to_shift = [0] + list(range(14, 32))
        for r in rows_to_shift:
            # Find the range of non-background pixels in this row
            non_bg = np.where(grid[r] != 10)[0]
            if len(non_bg) > 0:
                start, end = non_bg[0], non_bg[-1]
                # Shift everything from start to end one position to the right
                # The value at 'start' becomes background
                # Values shift: new[i+1] = old[i]
                for c in range(end, start - 1, -1):
                    if c + 1 < W:
                        new_grid[r, c + 1] = grid[r, c]
                new_grid[r, start] = 10
        return new_grid
    return grid

def is_level_complete(grid):
    # Win state not provided, but usually it involves reaching a target or clearing something.
    # In these games, often color 0 (black) or specific patterns are targets.
    # Since we don't have a win state, return False unless a known condition is met.
    return False