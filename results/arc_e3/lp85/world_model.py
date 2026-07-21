def engine(grid, action, data):
    """
    Updates the grid based on the observed transitions.
    The logic handles two main patterns:
    1. Filling column 0 in blocks of 5 with 14s and cycling values in row 19, cols 12-14.
    2. Filling column 0 (rows 0-3) with 5s and updating row 1, cols 10-13 to 14s.
    """
    # Case 4: Check if row 1, cols 10-13 are 5s, triggering a specific update.
    if len(grid) > 1 and len(grid[0]) > 13 and grid[1][10] == 5:
        # Update row 1, cols 10-13 to 14
        for c in range(10, 14):
            grid[1][c] = 14
        # Update column 0, rows 0-3 to 5
        for r in range(0, 4):
            grid[r][0] = 5
        return grid

    # Cases 0-3: Fill column 0 in blocks of 5 and cycle values in row 19.
    # 1. Find the first block of 5 cells in column 0 that are 0 and change them to 14.
    for r in range(len(grid) - 4):
        if all(grid[r + i][0] == 0 for i in range(5)):
            for i in range(5):
                grid[r + i][0] = 14
            break

    # 2. Update the cells in row 19, cols 12-14 based on the sequence [1, 2, 10, 9, 15].
    if len(grid) > 19 and len(grid[0]) > 14:
        seq = [1, 2, 10, 9, 15]
        current_val = grid[19][12]
        try:
            idx = seq.index(current_val)
            # Cycle to the next value in the sequence.
            next_val = seq[idx + 1] if idx + 1 < len(seq) else seq[0]
            for c in range(12, 15):
                grid[19][c] = next_val
        except ValueError:
            # If the current value is not in the sequence, no change is made to row 19.
            pass

    return grid

def is_level_complete(grid):
    """
    The level is considered complete when the entire column 0 is filled with 14s.
    """
    if not grid or not grid[0]:
        return False
    return all(grid[r][0] == 14 for r in range(len(grid)))