import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to involve moving blocks of colors (specifically 5, 9, 2)
    # based on actions. ACTION2 moves things right/down?
    # ACTION4 moves things left/up?
    # ACTION6 is click.
    
    # Based on the transitions, ACTION2 shifts patterns of color 5 and 2
    # # In each transition, we see changes at r8c14, r8c20 etc.
    # # Then r8c20, r8c26... then r8c26, r8c32...
    # # la own pattern shift.
    
    # Let's try to actually implement a movement logic for these blocks.
    # We identify "blocks" of non-zero cells.
    # Find all coordinates of color 5 or 2 or 9 or 8.
    # a few specific movements observed:
    # ACTION2: Shifts some regions right by 6 columns.
    #<|channel>thought
    # ACTION4: Changes some region from 5 to 9 in cols 14-20.
    # ACTION2 also affects rows 14-18, 20-24, 26-30, 32-36, 38-42, 44-48, 50-54.
    # These are groups of 5 rows.
    # The same column ranges (14-19, 20-25) are being modified.
    # It looks like the action is shifting a state between two colors (5 and 2).
    # In each transition for ACTION2, we see it changing values at c14-19 to 5 and c20-25 to 2.
    # Then next time, c20-25 to 5 and c26-31 to 2.
    # This suggests a "sliding" window of size 6.
    
    # Let's define the row blocks:
    # B1: 8-12, B2: 14-18, B3: 20-24, B4: 26-30, B5: 32-36, B6: 38-42, B7: 44-48, B8: 50-54.
    # All these blocks have height 5.
    # For ACTION2:
    # Transition 1: r8-12 shift? No, r8c14=5, r8c20=2.
    # Transition 2: r8-12 shift right by 6. r8c20=5, r8c26=2.
    # Transition 3: r8-12 shift right by 6 again. r8c26=5, r8c32=2.
    # Transition 4: r8-12 shift right by 6 again. r8c32=5, r8c32=5... wait.
    # Also in Trans 1: r14-18 change to 9.
    # In Trans 2: r20-24 change to 9.
    # In Trans 3: r26-30 change to 9.
    # In Trans 4: r32-36 change to 9.
    # This is a sequence of "activation" moving down the row blocks.
    
    # Let's implement this logic.
    new_grid = grid.copy()
    if action == 2:
        # Shift top block (r8-12) right by 6 columns.
        # We need to find where the '5's are and move them.
        for r in range(8, 13):
            row_vals = grid[r, 14:40]
            # Find first occurrence of color 5 in the window
            idx = np.where(grid[r, 14:40] == 5)[0]
            if len(idx) > 0:
                first_5 = idx[0]
                # Move it right by 6
                # The observed delta shows c14->c20, c20->c26...
                # So we shift the pattern [5x5, 2x5] right by 6.
                # But wait, the delta says r8c14:5x5, r8c20:2x5.
                # This means at t=0, r8c14..18=5, r8c20..24=2.
                # At t=1, r8c20..24=5, r8c26..30=2.
                # Let's just apply a simple shift to the region [14, 40].
                region = grid[r, 14:40].copy()
                shifted = np.roll(region, 6)
                new_grid[r, 14:40] = shifted

        # Also activate the next row block (B1-B8).
        # Find which block is currently '9'.
        current_block_idx = -1
        for b_start in [14, 20, 26, 32, 38, 44, 50]:
            if np.any(grid[b_start:b_start+5, 14:20] == 9):
                current_block_idx = b_start
                break
        
        next_block_start = -1
        blocks = [14, 20, 26, 32, 38, 44, 50]
        for i in range(len(blocks)):
            if blocks[i] == current_block_idx:
                if i + 1 < len(blocks):
                    next_block_start = blocks[i+1]
                break
        if current_block_idx == -1: # Start with first block
            next_block_start = 14
            
        if next_block_start != -1:
            # Set a pattern of 9s in the next block.
            # The delta shows r14c14:9x5 (rows 14-18).
            for r in range(next_block_start, next_block_start + 5):
                new_grid[r, 14:19] = 9
                # Also need to handle the "clearing" of the previous block' same region?
                # Delta for Trans 2 says r14c14:5x5... so it changes from 9 back to 5.
                # Let's find the block that was 9 and change it to 5.
                if current_block_idx != -1:
                    for r_prev in range(current_block_idx, current_block_idx + 5):
                        new_grid[r_prev, 14:19] = 5

    elif action == 4:
        # ACTION 4 seems to be a specific toggle or shift.
        # Transition: r50c14:5x5, r50c20:9x5 ...
        # This looks like changing color 5 to 9 in row block B8 (50-54) at c20-25.
        for r in range(50, 55):
            new_grid[r, 14:19] = 5
            new_grid[r, 20:25] = 9

    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing something or reaching a target.
    # In this game, maybe when all blocks are processed?
    # Or check if the bottom right cell (63, 63) is a certain color.
    # The delta shows r63c61:1x1, r63c60:1x1... it's counting down.
    # Let's assume win when grid[63, 63] is not 9.
    return False # Not enough info for a reliable win condition.

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for g50t is that the grid contains only 
    one color (excluding background 0).
    """
    grid = np.array(grid)
    non_zero_elements = grid[grid != 0]
    if non_zero_elements.size == 0:
        return False
    
    # Check if all non-zero elements are the same color
    return np.all(non_zero_elements == non_zero_elements[0])
