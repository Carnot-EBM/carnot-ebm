import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Right/Down? No, it seems to be a specific movement or shift.
    # ACTION2: Left/Up?
    # ACTION4: A jump or teleportation of blocks.
    # ACTION6: Click.
    
    # Based on the same pattern in deltas, we are simulating a puzzle game.
    # The cells changed by ACTION4 are symmetric across rows 48-49 and 59-60.
    # ACTION1 and ACTION2 seem to actually move pieces within the region r52-r56.
    # ACTION4 moves "holes" (color 0) at r48, r49, r59, r60.
    
    # Let's implement a logic where ACTION4 shifts the holes horizontally.
    # Shift holes from current positions to new positions.
    # Shift distance = 5 columns.
    # Shift direction = Right.
    # ACTION1 shifts something right.
    # ACTION2 shifts something left.
    # ACTION_C is not provided as data for ACTION6.
    # ACTION1/2/3/4/5 are keyboard actions.
    # ACTION4: Shifts holes (color 0) at r48, r49, r59, r60//
    # ACTION1: Shifts content in r52-r56 area.
    # ACTION2: Shifts content in r52-r56 area.
    # ACTION3: No observed transitions for ACTION3.
    # ACTION5: No same pattern.
    # ACTION6: Click.
    
    new_grid = grid.copy()
    
    if action == 4:
        # Move holes (color 0) and fill them with color 3.
        # Find all cells of color 0 in rows 48, 49, 59, 60.
        # For each cell of color 0, move it to the next position (col + 7).
        # The shift distance is 5 columns.
        # Shift amount = 5.
        shift = 5
        for r in [48, 49, 59, 60]:
            # find zeros
            zeros = np.where(grid[r] == 0)[0]
            for c in zeros:
                # Fill old zero with 3
                new_grid[r, c] = 3
                # Place new zero at col + 5 or wrap around?
                # In deltas, r48c15:3x5 means row 48, col 15 becomes 3 for 5 cols.
                # r48c22:0x5 means row 48, col 22 becomes 0 for 5 cols.
                # if c <= 63:
                #     if c < 63:
                #     # a a a a a
    
    # Since we are not provided with enough data to actually implement the complex logic,
    # and we can<|channel>thought
    # We will return the grid as is unless it's ACTION4 which shifts holes.
    # Let's try to be more general based on the observed transitions.
    
    # The most consistent pattern is that ACTION4 moves blocks of color 0 (holes)
    # horizontally by 7 columns each time.
    # shift_amount = 7
    # shift_direction = 1 # Right
    #
    # For rows [48, 49, 59, 60], find zeros.
    # Find contiguous blocks of zeros.
    # Find the first block of zeros in these rows.
    # Move them to current_col + 7.
    
    if action == 4:
        shift = 7
        for r in [48, 49, 59, 60]:
            # Find all indices where value is 0
            zeros = np.where(grid[r] == 0)[0]
            if len(zeros) > 0:
                start_col = zeros[0]
                end_col = zeros[-1]
                width = end_col - start_col + 1
                # Fill old position with 3
                new_grid[r, start_col:end_col+1] = 3
                # Place new zero at shifted position
                new_pos = (start_col + shift) % 64
                # Handle wrap around for a block of width 'width'
                for i in range(width):
                    new_grid[r, (new_pos + i) % 64] = 0
    
    # ACTION1 and ACTION2 move pieces in the region r52-r56.
    # They seem to be shifting content horizontally.
    # Let's implement simple shifts for them.
    if action == 1:
        # Shift right?
        shift = 1
        for r in range(52, 57):
            row = grid[r].copy()
            new_grid[r] = np.roll(row, shift)
    elif action == 2:
        # Shift left?
        shift = -1
        for r in range(52, 57):
            row = grid[r].copy()
            new_grid[r] = np.roll(row, shift)

    # The cell r63c63 is changing from 4 to something else or moving.
    # In deltas: r63c62:4x1 -> r63c61:4x1 -> r63c60:4x1 -> r63c59:4x1.
    # This happens on every ACTION4 (and some others).
    # It moves left by 1 each time.
    if action == 4:
        # Find the position of color 4 in row 63.
        pos = np.where(grid[63] == 4)[0]
        if len(pos) > 0:
            p = pos[0]
            new_grid[63, p] = 3 # Assuming background is 3
            new_grid[63, (p - 1) % 64] = 4
            
    return new_grid

def is_level_complete(grid):
    # No win state provided, but typically it's when a target is reached.
    # Based on the movement of the '4' token in row 63, maybe it reaches col 0?
    # Or holes reach a certain point.
    # For now, return False as no win condition was observed.
    return False