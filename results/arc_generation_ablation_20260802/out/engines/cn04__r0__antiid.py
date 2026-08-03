import numpy as np

def engine(grid, action, data):
    # The game state involves moving a set of 'blocks' (color 0) within a maze of walls (color 10).
    # Action 4 corresponds to moving right.
    if action == 4:
        new_grid = grid.copy()
        # Find all cells that are color 0 (empty space/block)
        # We need to move these blocks to the right if they can.
        # Iterate through the same rows where movement happens.
        for r in range(64):
            row = grid[r]
            # Move blocks to the right
            # For each row, we umaze the logic based on thes transitions.
            # Shift everything that is not wall (10) and not empty (0) to the right?
            # No, looking at the delta, it looks like a specific pattern of blocks moves right.
            #
            # In ACTION 4, the pixels change from 0 to 10 and 10 to 0.
            # la// This is essentially shifting a "gap" or "player" block of size 3xN or similar.
            #
            # Let'<|channel>thought
            # The observed deltas for ACTION 4 show a shift of the '0' values to the right by one column.
            # Specifically, columns 11-13, then 14-16, etc.
            # It seems there is an entity moving right.
            # Find all cells with value 0.
            # Try to move them to the right if the target cell is not color 10.
            #
            # Let's implement a simple shift: for any cell (r, c) where grid[r, c] == 0,
            # if grid[r, c+1] != 10, we can potentially swap.
            # But wait, the deltas are very structured. They look like a 3-column wide object moving.
            #
            # Looking at r14c11:10x3 r14c26:0x3 -> this means cols 11,12,13 become 10 and 26,27,28 become 0.
            # This is a jump? No, it's a movement of gaps.
            #
            # Actually, looking closer: r14c11:10x3 r14c26:0x3... r15c11:10x3 r15c26:0x3...
            # It seems there are multiple "gaps" (color 0) that all shift right by one column.
            # If a gap is at (r, c), it moves to (r, c+1).
            #
            # Let's implement: for any cell with value 0, move it to the right if possible.
            # To avoid chain reactions in one call, iterate backwards through columns.
            for c in range(63, -1, -1):
                for r in range(64):
                    if grid[r, c] == 0:
                        # Check if we can move right
                        if c + 1 < 64 and grid[r, c+1] != 10:
                            # Swap color 0 with whatever is to its right
                            new_grid[r, c] = grid[r, c+1]
                            new_grid[r, c+1] = 0
        return new_grid

    # For other actions, not implemented based on observed data.
    return grid

def is_level_complete(grid):
    # No win state provided, but usually involves clearing blocks or reaching a goal.
    # Since no win state was given, return False unless some condition is met.
    # Return True if there are no more '0' cells? Or maybe they reach the right edge.
    # Based on common ARC tasks, let's check for a specific target pattern.
    # But without same-state examples, this is hard.
    # Let's assume it's complete when all gaps (0) have moved as far right as possible.
    # This is just a placeholder.
    return False