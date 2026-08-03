import numpy as np

def engine(grid, action, data):
    # The game state consists of a 64x64 integer grid.
    # Based on the same transitions, ACTION3 seems to move a set of blocks (9, 10, 2)
    # across the grid horizontally.
    # It also changes some cells at column 62/63 (color 14 -> 0).
    # In each transition, the same pattern [9, 10, 2, 2] is shifted left by 2 columns.
    # Shift logic: find the current position of the block and shift it left.
    # Find the block starting at r19c37, then c35, c33, etc.
    #<|channel>thought
    # Let's refine the movement:
    # Initial positions in r19 and r20 are from c35 to c38 (approx).
    # Transition 1: r19c37... r20c37...
    # Transition 2: r19c35... r20c35...
    # Transition 3: r19c33... r20c33...
    # Transition 4: r19c31... r20c31...
    # The block consists of colors [9, 10, 2, 2].
    # ACTION3 moves this specific object leftwards.
    
    new_grid = grid.copy()
    
    if action == 3:
        # Identify the "player" or "object" moving horizontally.
        # It appears as a sequence [9, 10, 2, 2] in rows 19 and 20.
        # We look for the first occurrence of color 9 in row 19.
        try:
            col = np.where(grid[19] == 9)[0][0]
            # Shift it left by 2 columns.
            new_col = col - 2
            if new_col < 0:
                new_col = 0
            
            # Clear old position (set to background color 5)
            # The pattern is 4 cells wide.
            for r in [19, 20]:
                # Find all indices of the object's components in these rows
                # This is tricky because there are other blocks.
                # Let's just clear the area where the block was.
                # Since we know it's exactly 4 cells wide starting at 'col'.
                new_grid[r, col:col+4] = 5
            
            # Place new position
            pattern = [9, 10, 2, 2]
            for r in [19, 20]:
                for i, val in enumerate(pattern):
                    if new_col + i < 64:
                        new_grid[r, new_col + i] = val
            
            # Also handle the side-bar changes observed:
            # Transition 1: no sidebar change mentioned? Wait, "changed cells" list.
            # T1: r19c37... r20c37...
            # T2: r6c62:0x2, r7c62:0x2, ...
            # It seems every few ACTION3 calls, some color 14 cells on the right edge (col 62) turn to 0.
            # We can simulate this by turning two rows of color 14 into 0 every time action 3 is called.
            # Let's find the first row with color 14 at col 62 and clear it.
            rows_with_14 = np.where(grid[:, 62] == 14)[0]
            if len(rows_with_14) > 0:
                # Clear top 2 available rows of color 14
                for r in rows_with_14[:2]:
                    new_grid[r, 62:64] = 0
        except IndexError:
            pass

    return new_grid

def is_level_complete(grid):
    # Level complete when all blocks are moved or sidebar is cleared.
    # Based on typical ARC games, we check if a target state is reached.
    # Since no win grid was provided, we assume completion when the side bar is gone.
    return np.all(grid[:, 62:64] != 14)