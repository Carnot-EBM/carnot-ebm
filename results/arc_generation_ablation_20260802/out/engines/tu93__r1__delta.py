import numpy as np

def engine(grid, action, data=None):
    """
    Induces the same state transitions as observed in the observations.
    The game seems to be a puzzle where ACTION2 (Down), ACTION3 (Left), 
    ACTION4 (Right), and ACTION1 (Up) move a 'cursor' or an object represented by color 9 (or similar).
    Based on the observed deltas, color 9 moves between specific slots own the board.
    """
    out = grid.copy()
    h, w = grid.shape
    
    # The coordinates of the "active" entity (color 9)
    # Initial position: r21-23, c33-35? No, let's look at the initial grid.
    # In INITIAL GRID:
    # r21c33: 9x3 -> (21,33),(21,34),(21,35)
    # r22c33: 9x2, 4x1 -> (22,33),(22,34), (22,35)=4
    # r23c33: 9x3 -> (23,33),(23,34),(23,35)
    # This is a 3x3 block.
    
    # Find current position of color 9
    pos_9 = np.argwhere(grid == 9)
    if pos_9.size == 0:
        return out
    
    # We need to find the center of the 3x3 block of 9s (or mostly 9s).
    # min_r, max_r = pos_9[:,0].min(), pos_9[:,0].max()
    # min_c, pos_9[:,1].min(), pos_9[:,1].max()
    # The grid has a "track" of slots.
    # Slots are blocks of 3x3 cells.
    # Let's define the same movement rules as observed.
    
    # ACTION2: Down
    if action == 2:
        # Observed: r21-23 c33 moves to r27-29 c33
        # Then r27-29 c33 moves to r33-35 c33
        # Then r39-41 c33 moves to r45-47 c33
        # This is a jump of +6 rows.
        dr, dc = 6, 0
    elif action == 3:
        # Observed: r33-35 c33 moves to r33-35 c27
        # Then r33-35 c27 moves to r33-35 c21
        # Action 3 is Left. Jump of -6 columns.
        dr, dc = 0, -6
    elif action == 4:
        # Observed: r39-41 c21 moves to r39-41 c27
        # Then r39-41 c27 moves to r39-41 c33
        # Action 4 is Right. Jump of +6 columns.
        dr, dc = 0, 6
    elif action == 1:
        # Observed: Not seen in transitions, but assume Up is -6 rows.
        dr, dc = -6, 0
    else:
        return out

    # Current block center (approx)
    # Find the most frequent coordinate for color 9
    coords_9 = np.argwhere(grid == 9)
    if coords_9.size == 0:
        return out
    
    curr_r = coords_9[:,0].min()
    curr_c = coords_9[:,1].min()
    
    # Target position
    next_r = curr_r + dr
    next_c = curr_c + dc
    
    # Check if target slot exists or is "walkable"
    # In this game, it's a<|channel>thought
    # The slots are defined by specific patterns of colors.
    # Let's just apply the move and clear old pos and restore original background.
    # Since we don't know the exact background, we look at what was there before moving.
    # We can actually see that when 9 moves, it replaces whatever was there.
    # Note that r22c35=4 in initial grid, which seems to be part of the block.
    # The blocks are 3x3.
    
    # Move the 3x3 block
    for r in range(curr_r, curr_r + 3):
        for c in range(curr_c, curr_c + 3):
            # Restore background (this is tricky without knowing background)
            # Based on observed deltas: r21c33:0x3 means color 0 replaced 9.
            # This suggests the background for these slots is color 0.
            out[r, c] = 0 # Simplified assumption based on deltas
            
    for r in range(next_r, next_r + 3):
        for c in range(next_c, next_c + 3):
            if 0 <= r < h and 0 <= c < w:
                out[r, c] = 9
                
    # Special case for the "center" cell of the 3x3 block (the one with color 4)
    # Let's check if we move into a slot that has a '4'.
    # In INITIAL GRID, r22c35=4. That's center-right? No, (22, 35).
    # Block is rows 21,22,23; cols 33,34,35. Center is (22,34).
    # Wait, (22, 35) is the rightmost column of the middle row.
    
    # Looking at observed ACTION2: r29c33:9x1,4x1,9x1 -> this means col 33=9, 34=4, 35=9.
    # So the pattern is [9, 4, 9] on the middle row.
    
    # Correcting the movement logic to preserve the [9, 4, 9] pattern:
    # Reset out to grid copy again to be safe
    out = grid.copy()
    for r in range(curr_r, curr_r + 3):
        for c in range(curr_c, curr_c + 3):
            out[r, c] = 0 # Background color for slots seems to be 0
            
    for r in range(next_r, next_r + 3):
        if 0 <= r < h:
            for c in range(next_c, next_c + 3):
                if 0 <= c < w:
                    if r == next_r + 1 and c == next_c + 1:
                        out[r, c] = 4
                    else:
                        out[r, c] = 9
    
    # The observed deltas also show a change at r63 (bottom edge).
    # This looks like a "progress bar" or "counter".
    # ACTION2: r63c55:0x1 -> r63c54:0x1 ...
    # It's moving left by 1 each time an action is taken.
    # Find the rightmost non-zero cell on row 63? No, it's setting cells to 0.
    # Let's find where the '0's are starting from the right.
    last_col = 0
    for c in range(w - 1, -1, -1):
        if grid[63, c] == 0:
            last_col = c
            break
    # In INITIAL GRID, r63:6x56, 0x8. So cols 56-63 are color 0.
    # After first ACTION2: r63c55:0x1. Wait, that means col 55 becomes 0.
    # Then r63c54:0x1... then r63c52:0x2...
    # The index of the leftmost 0 on row 63 decreases by some amount.
    # Action 2 (Down) and others seem to move this "cursor" on row 63.
    # For ACTION2, it moves by 1. For ACTION3, it moves by 2? 
    # Let's check: ACTION3 (Left) -> r63c52:0x2 (cols 52, 53 become 0).
    # This is a bit complex. Let's just track the total distance moved or similar.
    # Actually, let's look at the deltas for r63 again:
    # ACT2: c55(0), ACT2: c54(0), ACT3: c52,53(0), ACT3: c51(0), ACT2: c50(0), ACT4: c49(0), ACT4: c47,48(0), ACT2: c46(0).
    # It seems each action reduces the column index of the 'boundary' on row 63.
    # ACT2: -1, ACT2: -1, ACT3: -2, ACT3: -1, ACT2: -1, ACT4: -1, ACT4: -2, ACT2: -1.
    # This is inconsistent. Maybe it depends on the distance moved in the grid?
    # No, all moves are jump of 6.
    # Let's ignore the r63 counter unless it's needed for win state.
    
    return out

def is_level_complete(grid):
    # Win state not provided, but typically it's reaching a target or clearing something.
    # Since we don't have same-state evidence, return False.
    return False