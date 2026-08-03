import numpy as np

def engine(grid, action, data):
    # The observed transitions show a pattern of shifting colors in blocks.
    # ACTION 0 seems to be a "cycle" or "shift" operation that affects multiple regions.
    # Based on the same-action repeated calls and the delta patterns, it's likely
    # that these are shifts of specific color groups across predefined positions.
    # Let's induce a general rule for the shift.
    
    # Define the target blocks (the coordinates of the cells being changed)
    # From the deltas, we identify several sets of columns/rows where changes occur.
    # # Region 1: Rows 19-22, Cols 12, 18, 24, 30, 36, 42, 48
    # # Region 2: Rows 25-28, Cols 12, 48
    # #<|channel>thought
    # # Region 3: Rows 31-34, Cols 12, 48
    # # Region 4: Rows 37-40, Cols 12, 48
    # # Region 5: Rows 43-46, Cols 12, 18, 24, 30, 36, 42, 48
    
    # The colors involved in the same regions are [1, 2, 9, 10, 11, 15]
    # In each transition, the values at these positions change to a new value from this set.
    # This is a permutation of existing values.
    
    # Define the blocks and their coordinates
    blocks = [
        # (row_start, row_end, col_start, width)
        (19, 23, 12, 4), (19, 23, 18, 4), (19, 23, 24, 4), (19, 23, 30, 4), (19, 23, 36, 4), (19, 23, 42, 4), (19, 23, 48, 4),
        (25, 29, 12, 4), (25, 29, 48, 4),
        (31, 35, 12, 4), (31, 35, 48, 4),
        (37, 41, 12, 4), (37, 41, 48, 4),
        (43, 47, 12, 4), (43, 47, 18, 4), (43, 47, 24, 4), (43, 47, 30, 4), (43, 47, 36, 4), (43, 47, 42, 4), (43, 47, 48, 4)
    ]
    
    # The observed ACTION 0 shifts the colors of these blocks.
    # Let's look at the values in Region 1 and Region 5.
    # In each transition, the same set of columns are shifted.
    # a = [v1, v2, v3, v4, v5, v6, v7] -> [v2, v3, v4, v5, v6, v7, v1]
    # This is a simple rotation.
    
    # Identify all cells that belong to any block.
    # new_grid = grid.copy()
    # if action == 0:
    #     for b in blocks:
    #         # Logic for rotating colors among blocks
    #         pass
    # return new_grid

    # Since we only have ACTION 0 and it seems to be a complex permutation,
    # let's try to implement the specific shift seen in the deltas.
    
    new_grid = grid.copy()
    if action == 0:
        # Shift logic based on observations
        # We can see that the first column (col 0) changes color from 14 to 5.
        # The blocks also change colors.
        # Let's define the regions and rotate their values.
        
        # Col 0 rotation: 14 -> 5 -> 14... or similar.
        # In the observed transitions, col 0 rows 0-4 become 5, then 5-9 become 5, then 10-14 become 5.
        # This looks like a "cursor" of color 5 moving down col 0.
        
        # For the blocks, they are shifting colors.
        # Region 1 (rows 19-22): [v1, v2, v3, v4, v5, v6, v7] shifts left by 1.
        # Region 5 (rows 43-46): [v1, v2, v3, v4, v5, v6, v7] shifts left by 1.
        
        # Update Col 0 cursor
        # Find where the '5's are in col 0
        fives = np.where(grid[:, 0] == 5)[0]
        if len(fives) > 0:
            # Reset current fives to 14
            new_grid[fives, 0] = 14
            # Move them down by 5
            start = fives[0] + 5
            if start < 64:
                new_grid[start : start+5, 0] = 5
        else:
            # Start at row 0
            new_grid[0:5, 0] = 5

        # Block rotations
        def rotate_block_values(r_start, r_end, cols):
            for r in range(r_start, r_end):
                vals = [grid[r, c] for c in cols]
                rotated = vals[1:] + [vals[0]]
                for i, c in enumerate(cols):
                    new_grid[r, c] = rotated[i] # This is a simplification; blocks are width 4
                    new_grid[r, c:c+4] = rotated[i]

        rotate_block_values(19, 23, [12, 18, 24, 30, 36, 42, 48])
        rotate_block_values(43, 47, [12, 18, 24, 30, 36, 42, 48])
        
        # The other regions (rows 25-28, 31-34, 37-40) only have two columns (12 and 48).
        # They seem to swap or shift based on the others.
        def rotate_two_blocks(r_start, r_end, col1, col2):
            for r in range(r_start, r_end):
                v1 = grid[r, col1]
                v2 = grid[r, col2]
                new_grid[r, col1:col1+4] = v2
                new_grid[r, col2:col2+4] = v1

        rotate_two_blocks(25, 29, 12, 48)
        rotate_two_blocks(31, 35, 12, 48)
        rotate_two_blocks(37, 41, 12, 48)

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a certain pattern is achieved.
    # Given the data, we can't induce a specific win condition.
    return False