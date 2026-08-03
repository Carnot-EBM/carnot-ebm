import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    if action == 4: # ACTION4 seems to be a rightward shift of some elements
        new_grid = grid.copy()
        # Identify regions that are not color 10 (the background)
        # The observed transitions show shifts in specific columns.
        # We need to find the same pattern of "non-background" cells and move them.
        # It looks like it's shifting a set of blocks of width 3.
        # Specifically, it's moving the 'active' part of the maze/structure.
        # Let's look at the<|channel>thought
        # In each transition, the column indices change by 3.
        # Find all cells that are NOT color 10.
        # non_bg//= 10
        # Actually, let's just implement a simple shift for any cell that isn't 10.
        # But wait, the delta shows only certain rows being affected.
        # Only rows 14-31 are shifted? No, row 0 is also changed.
        # Row 0 has r0c16:0x1, then r0c17:0x1... this is a single pixel moving.
        # Let's try a general rule: if action == 4, shift everything that is not 10 one unit right.
        # However, the deltas show jumps of 3 pixels (e.g., c11 -> c14 -> c17).
        # This suggests ACTION4 moves "blocks" or shifts the entire structure by 3 units.
        # The observed data shows columns shifting by 3.
        # Let's find all coordinates where grid[r, c] != 10 and move them to [r, c+3].
        # We must be careful about boundaries.
        
        # To avoid overwriting, we create a new grid of background color 10.
        # First, identify what should actually be moved.
        # In the provided transitions, it seems like ONLY cells in specific ranges are moving.
        # But let's try shifting all non-background cells by 3.
        
        # Wait, looking at the delta again:
        # Transition 1: r14c11:10x3, r14c26:0x3 ...
        # Transition 2: r14c14:10x3, r14c29:0x3 ...
        # It looks like it's swapping colors? No, it's replacing values.
        # Actually, if you look closely at the deltas:
        # r14c11 becomes 10 (bg), while r14c26 becomes 0.
        # Then r14c14 becomes 10, while r14c29 becomes 0.
        # This is exactly a shift of 3 pixels to the right for the "non-10" parts.
        
        # Let's implement: Shift all non-10 cells 3 units to the right.
        # And since row 0 also changes (r0c16:0x1 -> r0c17:0x1), that's a shift of 1 unit.
        # This is contradictory. Let's re-examine row 0.
        # Trans 1: r0c16:0x1 (cell 16 becomes 0)
        # Trans 2: r0c17:0x1 (cell 17 becomes 0) - wait, no, this is not how run-length works.
        # "changed cells... = r0c16:0x1" means cell (0, 16) now has value 0.
        # "changed cells... = r0c17:0x1" means cell (0, 17) now has value 0.
        # If it was shifting, then (0, 16) should have become 10 and (0, 17) should have become 0.
        # But only one cell is listed as changed per transition for row 0.
        # This means the previous change persists? No, deltas are applied to the prior grid.
        # So if r0c16 became 0 in T1, and r0c17 becomes 0 in T2, then both are 0?
        # That doesn't sound like a shift.
        
        # Let's look at the structure again. The background is 10.
        # Most of the maze is color 0 or 8 or 14.
        # ACTION4 shifts the non-background parts by 3 units right.
        # For row 0, maybe it's just marking a path?
        # Actually, let's try this: Shift all cells that are NOT 10 three pixels to the right.
        # And set their old positions to 10.
        
        res = np.full_like(grid, 10)
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] != 10:
                    nc = c + 3
                    if nc < grid.shape[1]:
                        res[r, nc] = grid[r, c]
                    else:
                        pass # Out of bounds
        
        # Special case for row 0 based on deltas:
        # T1: r0c16:0x1 -> cell (0,16)=0
        # T2: r0c17:0x1 -> cell (0,17)=0
        # This looks like a "brush" moving right one pixel at a time.
        # But ACTION4 is called multiple times.
        # Let's check if there's any other action. No, only ACTION4.
        # If we shift by 3, then Row 0 should also shift by 3?
        # The delta says r0c16:0x1, then r0c17:0x1... that's a shift of 1.
        # Maybe the rule is: Shift everything non-10 by 3, EXCEPT row 0 which shifts by 1?
        
        # Let's refine:
        final_grid = np.full_like(grid, 10)
        for r in range(grid.shape[0]):
            shift = 3 if r != 0 else 1
            for c in range(grid.shape[1]):
                if grid[r, c] != 10:
                    nc = c + shift
                    if nc < grid.shape[1]:
                        final_grid[r, nc] = grid[r, c]
        return final_grid

    return grid

def is_level_complete(grid):
    # Win state usually involves clearing something or reaching a goal.
    # Without a win state example, we assume it's complete when some condition is met.
    # In many ARC games, this might be all cells of one color being gone or a specific pattern.
    # Given the data, let's just return False as we don't have a target.
    return False