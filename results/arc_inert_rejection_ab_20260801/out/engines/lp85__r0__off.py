import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of shifting colors in specific blocks.
    # Action 0 seems to trigger these shifts.
    # Based on the delta, it's the same set of cells being updated repeatedly.
    # We can observe that the same columns (12, 18, 24, 30, 36, 42, 48) are involved.
    # These are multiples of 6.
    # Let's define the regions and the shift logic.
    
    if action != 0:
        return grid.copy()
    
    new_grid = grid.copy()
    
    # Update column 0 based on some sequence
    # Transition 1: r0-r4 c0 -> 5
    # Transition 2: r5-r9 c0 {already 5?} no, r5-r9 c0 -> 5
    #<|channel>thought
    # It looks like a cursor or marker moving down column 0.
    # Count how many 5s are already there.
    count_5s = np.sum(new_grid[:, 0] == 5)
    start_row = count_5s
    # The observed transitions show blocks of 5 rows each.
    # If we have 0 fives, start at row 0.
    # If we have 5 fives, start at row 5.
    # If we have 10 fives, start at row 10.
    # This is consistent with the delta: r0c0:5x1, then r5c0:5x1, then r10c0:5x1.
    if start_row < 64:
        new_grid[start_row : start_row + 5, 0] = 5
    
    # Now handle the block shifts.
    # The columns involved are [12, 18, 24, 30, 36, 42, 48].
    # These correspond to indices in a list of colors.
    # Let's identify the colors being shifted.
    # In ACTION0 (level 0->0), first transition:
    # r19-r22 c12: 2, c18: 10, c24: 9, c30: 15, c36: 11, c42: 2, c48: 15
    # Second transition:
    # r19-r22 c12: 10, c18: 9, c24: 15, c30: 11, c36: 2, c42: 15, c48: 9
    # Third transition:
    # r19-r22 c12: 9, c18: 15, c24: 11, c30: 2, c36: 15, c42: 9, c48: 10
    # This is a cyclic shift of values across these columns for specific row blocks.
    
    # Row blocks involved: [19-22], [25-28], [31-34], [37-40], [43-46]
    row_blocks = [(19, 22), (25, 28), (31, 34), (37, 40), (43, 46)]
    cols = [12, 18, 24, 30, 36, 42, 48]
    
    for rb in row_blocks:
        r_start, r_end = rb
        # Get current colors at the first column of the block
        current_colors = []
        for c in cols:
            current_colors.append(grid[r_start, c])
        
        # Shift them left by one position?
        # Let's check Transition 1 -> 2 for r19-22:
        # T1: [2, 10, 9, 15, 11, 2, 15]
        # T2: [10, 9, 15, 11, 2, 15, 9]
        # This is a shift: index i becomes index i-1.
        # New values: [C1, C2, C3, C4, C5, C6, C0]
        shifted_colors = current_colors[1:] + [current_colors[0]]
        
        # Apply to all rows in the block
        for i, c in enumerate(cols):
            new_grid[r_start : r_end + 1, c] = shifted_colors[i]

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it involves filling something or reaching a pattern.
    # Given the data, we can't induce a specific win condition.
    # Return False as default unless a clear target is found.
    return False