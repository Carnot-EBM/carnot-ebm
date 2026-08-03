import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the same pattern of colors 
    # changes in blocks based on some trigger or sequence.
    # Based on the observed transitions, ACTION0 rotates or shifts 
    # the values in specific rectangular regions.
    # These regions are defined by rows and columns indices own from the INITIAL GRID.
    # In thes observations, we see that the same set of cells change their values.
    #<|channel>thought
    # Let's define the coordinates of the changing blocks.
    # From the delta, we observe that column 0 always changes (r0-r4, then r5-r9, then r10-r14).
    # This looks like a cycle of 5 rows in column 0.
    # And there are several blocks of size 4x4.
    # Looking at the deltas, the values within these 4x4 blocks are shifting/rotating.
    # Specifically, let's identify the 4x4 blocks:
    # Block 1: r19-22, c12-15
    # Block 2: r19-22, c18-21
    # Block 3: r19-22, c24-27
    # Block 4: r19-22, c30-33
    # Block 5: r19-22, c36-39
    # Block 6: r19-22, c42-45
    # Block 7: r19-22, c48-51
    # Block 8: r25-28, c12-15
    # Block 9: r25-28, c48-51
    # Block 10: r31-34, c12-15
    # Block 11: r31-34, c48-51
    # Block 12: r37-40, c12-15
    # Block 13: r37-40, c48-51
    # Block 14: r43-46, c12-15
    # Block 15: r43-46, c18-21
    # Block 16: r43-46, c24-27
    # Block 17: r43-46, c30-33
    # Block 18: r43-46, c36-39
    # Block 19: r43-46, c42-45
    # Block 20: r43-46, c48-51
    
    # Now let's check the value shifts.
    # In ACTION0 (first), Block 1 changes from 1x4 to 2x4.
    # Wait, the run length "2x4" means color 2 for 4 cells.
    # Let's look at the values in these blocks across transitions.
    # The colors involved are [1, 2, 9, 10, 11, 15].
    # It looks like a permutation of these colors is applied to all active blocks.
    # Color sequence: 1 -> 2 -> 10 -> 1 -> ... ? No.
    # Let's trace Block 1 (r19-22, c12-15):
    # Initial: 1
    # Action 0(1): 2
    # Action 0(2): 10
    # Action 0(3): 9  Wait, no.
    # Let's re-examine:
    # Transition 1: r19c12: 2x4
    # Transition 2: r19c12: 10x4
    # Transition 3: r19c12: 9x4
    # Sequence for Block 1: 1 -> 2 -> 10 -> 9
    # Trace Block 2 (r19-22, c18-21):
    # Initial: 4? No, INITIAL GRID says r19: 14x1, 4x11, 1x4, 4x2, 2x4...
    # Col indices: 0..13, 14..24, 25..28, 29..30, 31..34...
    # So Block 1 is at col 25-28.
    # Let's just use the deltas to find the mapping.
    # The colors are shifting in a cycle.
    # For any cell that changes, its new value depends on its old value.
    # Looking at ACTION0(1) delta:
    # r19c12 (was 4?) -> 2
    # r19c18 (was 4?) -> 10
    # r19c24 (was 4?) -> 9
    # This is confusing because I don't have the grid values.
    # But wait, if we look at all cells changing in one action, they all change together.
    # Let's assume there's a global color map for ACTION0.
    # Map: {old_val: new_val}
    # From Transition 1:
    # Color at r19c12 was X, becomes 2.
    # Color at r19c18 was Y, becomes 10.
    # Color at r19c24 was Z, becomes 9.
    # In Transition 2:
    # Color at r19c12 (now 2) becomes 10.
    # Color at r19c18 (now 10) becomes 9.
    # Color at r19c24 (now 9) becomes 15.
    # So: 2 -> 10, 10 -> 9, 9 -> 15.
    # In Transition 3:
    # Color at r19c12 (now 10) becomes 9.
    # Color at r19c18 (now 9) becomes 15.
    # Color at r19c24 (now 15) becomes 11.
    # Sequence: ... -> 2 -> 10 -> 9 -> 15 -> 11 -> ...
    # Let's check other colors: 1 is also there.
    # Transition 1: r25c12 (was X) -> 1.
    # Transition 2: r25c12 (now 1) -> 2.
    # Transition 3: r25c12 (now 2) -> 10.
    # So the sequence is: 1 -> 2 -> 10 -> 9 -> 15 -> 11 -> 1...
    # Wait, let's re-verify:
    # Trans 1: r25c12 becomes 1.
    # Trans 2: r25c12 becomes 2.
    # Trans 3: r25c12 becomes 10.
    # This matches! 1 -> 2 -> 10.
    # Now let's find the full cycle: 1 -> 2 -> 10 -> 9 -> 15 -> 11 -> 1?
    # Let's check if any others exist. The observed values are [1, 2, 9, 10, 11, 15].
    # That's 6 colors. Cycle length 6.
    # Map: {1:2, 2:10, 10:9, 9:15, 15:11, 11:1}
    
    # Now we need to know WHICH cells change.
    # Column 0 changes in blocks of 5 rows: (0-4), (5-9), (10-14).
    # And there are these 4x4 blocks.
    # But wait, the problem says "Induce SIMPLE GENERAL rules".
    # Maybe ACTION0 just shifts ALL cells that have a color from this set {1, 2, 9, 10, 11, 15}?
    # Let's test this hypothesis.
    # In Transition 1, r0c0 becomes 5. Wait, 5 is not in our set.
    # r0c0 was 14. It becomes 5.
    # r5c0 was 14. It becomes 5.
    # r10c0 was 14. It becomes 5.
    # So 14 -> 5?
    # Let's check if 5 changes back.
    # No, it doesn't seem to.
    # Actually, look at the deltas again.
    # Trans 1: r0-r4 c0 become 5.
    # Trans 2: r5-r9 c0 become 5.
    # Trans 3: r10-r14 c0 become 5.
    # This is a separate mechanism. Every time ACTION0 is pressed, the next block of 5 rows in col 0 turns to color 5.
    
    # Now let's refine the color shift for the other cells.
    # The cells that change are those with colors {1, 2, 9, 10, 11, 15}.
    # Let's see if ALL such cells change.
    # In Transition 1:
    # r19c12 (was something) -> 2
    # r19c18 (was something) -> 10
    # ... and so on.
    # If we apply the map {1:2, 2:10, 10:9, 9:15, 15:11, 11:1} to all cells with these colors, does it match?
    # Let's check Block 1 again.
    # Initial Grid: r19 has "1x4" at some point. That's color 1.
    # Trans 1: r19c12 becomes 2. (1 -> 2). Matches!
    # Trans 2: r19c12 becomes 10. (2 -> 10). Matches!
    # Trans 3: r19c12 becomes 9. (10 -> 9). Matches!
    # This is a very simple general rule!
    
    # Final Rules for ACTION0:
    # 1. Shift colors in set {1, 2, 9, 10, 11, 15} using map {1:2, 2:10, 10:9, 9:15, 15:11, 11:1}.
    # 2. Change the next block of 5 rows in column 0 from 14 to 5.
    # The blocks are (0-4), (5-9), (10-14), ...
    # We need to keep track of which block was last changed.
    # But wait, the `engine` function must be pure. It doesn't have state other than the grid.
    # How can we know which block to change?
    # Look at the grid: the cells in col 0 that are already color 5 tell us where we are!
    # If rows 0-4 are 5, then the next block is 5-9.
    # If rows 0-9 are 5, then the next block is 10-14.
    # This makes it pure!

    new_grid = grid.copy()
    
    if action == 0:
        # Color shift
        mapping = {1: 2, 2: 10, 10: 9, 9: 15, 15: 11, 11: 1}
        for r in range(new_grid.shape[0]):
            for c in range(new_grid.shape[1]):
                val = new_grid[r, c]
                if val in mapping:
                    new_grid[r, c] = mapping[val]
        
        # Column 0 update
        # Find how many blocks of 5 are already color 5
        count_5 = 0
        for r in range(0, 64, 5):
            if np.all(grid[r:r+5, 0] == 5):
                count_5 += 1
            else:
                break
        
        start_row = count_5 * 5
        if start_row + 5 <= 64:
            new_grid[start_row:start_row+5, 0] = 5

    return new_grid

def is_level_complete(grid):
    # Usually a level is complete when some target state is reached.
    # In this game, maybe it's when all the col 0 blocks are filled?
    # Or when certain colors are aligned?
    # Since we don't have a win state grid, let's guess based on common ARC patterns.
    # Let's assume it's complete when column 0 has 13 blocks of 5 (up to row 64).
    # But wait, 13*5 = 65, and the grid is 64x64. So 12 blocks (up to 60).
    # Let's check if there's any other obvious condition.
    # Given the limited info, let's use a simple condition.
    # Actually, most ARC games end when a specific pattern is formed.
    # Without more data, this is a guess.
    return np.all(grid[0:60, 0] == 5)