import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action == 4: # ACTION4 seems to be a rightward shift of some elements
        new_grid = grid.copy()
        # Identify all cells that are not color 10 (the background/wall)
        # The observed transitions show shifts in specific regions.
        # It looks like it's shifting a "window" or "cursor" of non-background cells.
        # Let's try a simple approach: find all cells that not equal to 10 and shift them.
        # But wait, the delta shows very specific patterns.
        # Looking at the<|channel>thought process, it's likely moving a set of blocks.
        # We need to identify what is being moved.
        # In each ACTION4 transition, the column indices increase by 3.
        # Shift everything that isn't 10? No, because only certain parts move.
        # Only columns 11 through 63 are affected in the first few steps.
        # Only rows 14 through 31 are shifted.
        # Only row 0 is changed slightly (a single cell changes from 10 to 0 or vice versa).
        # Let's refine this:
        # Find the leftmost column index 'c' where any cell in rows 14-31 is not 10.
        # Find the rightmost column index 'c_max' where any cell in rows 14-31 is not 10.
        # The shift is by 3 units.
        # The observed transitions show that if a cell was non-background, it becomes background,
        # and the cell 3 units to its right becomes non-background.
        # This looks like shifting a "pattern" of colors.
        # Specifically, let's look at the delta for r14c11:10x3. That means cells (14, 11), (14, 12), (14, 13) become color 10.
        # Then r14c26:0x3 means cells (14, 26), (14, 27), (14, 28) become color 0.
        # Wait, the ACTION4 shifts are actually moving the "empty" spaces (color 0) or specific blocks.
        # Let's re-examine: r14c11:10x3 (cells 11,12,13 -> 10) and r14c26:0x3 (cells 26,27,28 -> 0).
        # In the initial grid, r14 has 0x15 starting at col 11. So cols 11-25 are 0.
        # After first ACTION4: cols 11-13 become 10, cols 26-28 become 0.
        # This is a shift of the "hole" (color 0) to the right by 3 units? No, 26-11 = 15.
        # The hole was 15 wide. Now it's shifted?
        # Actually, look at the delta again: r14c11:10x3 means cells 11,12,13 become 10.
        # And r14c26:0x3 means cells 26,27,28 become 0.
        # Initial state r14: 10x11, 0x15, 10x38. Hole is from 11 to 25.
        # First action: 11-13 becomes 10, 26-28 becomes 0. New hole: 14 to 28.
        # Second action: 14-16 becomes 10, 29-31 becomes 0. New hole: 15 to 31.
        # It seems ACTION4 shifts the "empty space" (color 0) one step (of size 3) to the right.
        # But wait, the shift is only for a small part of the hole.
        # Let's simplify: ACTION4 moves all non-background elements in rows 14-31 and row 0 by some amount?
        # No, it looks like it's shifting the entire pattern of colors [0, 4, 8, 10, 14] within those rows.
        # Let's try this: find the first column 'c' where grid[row][c] != 10.
        # Shift everything that isn't 10 to the right by 3? No, the delta says r14c11:10x3.
        # That means cells at col 11, 12, 13 become 10.
        # And r14c26:0x3 means cells at col 26, 27, 28 become 0.
        # This is exactly moving the boundary of the color-0 region.
        # The color-0 region starts at c_start and ends at c_end.
        # After action: c_start = c_start + 3, c_end = c_end + 3.
        # But only for certain rows.
        # Let's look at the "hole" in each row from 14 to 31.
        # Row 14: hole is cols 11-25.
        # Row 17: holes are 11-13 and 23-25.
        # Action 4 shifts these boundaries.
        # For any cell (r, c) where grid[r][c] == 0, if it's on the left edge of a 0-block, it becomes 10?
        # No, let's just implement the shift:
        # For rows 14-31:
        #   Find all contiguous blocks of color 0.
        #   For each block [c_start, c_end]:
        #     Set cells [c_start, c_start + 2] to 10.
        #     Set cells [c_end + 1, c_end + 3] to 0.
        # Also handle row 0: r0c16:0x1 -> r0c17:0x1 ... this is shifting a single 0 to the right by 1.
        
        new_grid = grid.copy()
        # Row 0 shift
        for c in range(63, -1, -1):
            if grid[0, c] == 0:
                new_grid[0, c] = 10
                if c + 1 < 64:
                    new_grid[0, c+1] = 0
                break # only one 0 in row 0

        # Rows 14-31 shift
        for r in range(14, 32):
            # Find all blocks of 0s
            blocks = []
            start = -1
            for c in range(64):
                if grid[r, c] == 0:
                    if start == -1: start = c
                else:
                    if start != -1:
                        blocks.append((start, c - 1))
                        start = -1
            if start != -1:
                blocks.append((start, 63))
            
            for b_start, b_end in blocks:
                # Shift the block by 3 units
                # The left 3 cells become background (10)
                for c in range(b_start, min(b_start + 3, 64)):
                    new_grid[r, c] = 10
                # The right 3 cells become 0
                for c in range(b_end + 1, min(b_end + 4, 64)):
                    new_grid[r, c] = 0
        return new_grid

    return grid

def is_level_complete(grid):
    # Win state is not provided, but usually it's when a certain pattern is reached.
    # Given the transitions, maybe it's when the "hole" reaches the end?
    # Or some other condition. Since we don't have the win state, let's return False.
    return False