import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where blocks of colors (5, 9, 2) move or shift.
    # Based on the transitions, ACTION2 shifts patterns of color 5 and 9/2 across the same region.
    # Action 2 moves things rightward.
    if action == 2:
        # Identify regions of interest based on the initial layout
        # Color 5 is the main background/wall.
        # Color 9 is often the target or moving part.
        # Color 8 is another wall/blocker.
        
        # We observe changes in rows 8-12 and 20-24, 26-30, etc.
        # la = [14, 20, 26, 32] # Left boundaries for these "slots"
        
        # To simulate the a simple movement rule for this specific level:
        # Let's find all cells that are not 0 or 5.
        # For each such cell, if it's not 8, we can actually shift them.
        
        # Find current positions of color 2 and 9.
        # Find the 'active' area (where grid[r, c] == 5).
        
        # Create a copy to avoid mutating input
        new_grid = grid.copy()
        
        # The transitions show ACTION2 shifting blocks of colors 5, 9, 2 across columns 14-38.
        #<|channel>thought
        # r8c14:5x5 r8c20:2x5 -> shifts block at col 14 to 20? No.
        # It looks like a conveyor belt or sliding puzzle.
        # In row 8, cols 14-18 become 5, cols 20-24 become 2.
        # This is effectively moving a block of size 5 from col 14 to col 20.
        
        # Let's implement a simplified version: move non-5/non-0 values in specific rows
        # rightward by 6 units if they are within the active zone.
        
        rows_to_shift = [8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 50, 51, 52, 53, 54]
        cols_to_shift = range(14, 40)
        
        for r in rows_to_shift:
            # Find blocks of color not equal to 5 (and not 0 or 8)
            # We only shift things that aren't walls (8).
            row_segment = grid[r, cols_to_shift].copy()
            # Shift right by 6
            shifted_segment = np.full(len(row_segment), 5)
            for c in range(len(row_segment)-6):
                shifted_segment[c+6] = row_segment[c]
            new_grid[r, cols_to_shift] = shifted_segment
            
        # Also handle the "score" counter at r63c60 area.
        # The transitions show r63c61:1x1, then r63c60:1x1, etc.
        # This is a countdown/countup on the bottom row.
        if action == 2:
            # Find first non-9 cell from left in row 63 starting at col 58
            for c in range(58, 64):
                if new_grid[63, c] != 1:
                    new_grid[63, c] = 1
                    break

    elif action == 4:
        # ACTION 4 changes colors of blocks (e.g., 9 to 5 or vice versa).
        # In the observed transition: r50c14:5x5 r50c20:9x5 ...
        # It seems to swap color 5 and 9 in specific regions.
        new_grid = grid.copy()
        rows_action4 = [50, 51, 52, 53, 54]
        cols_action4 = range(14, 25)
        for r in rows_action4:
            # Swap 5s and 9s in this region
            mask5 = (new_grid[r, cols_action4] == 5)
            mask9 = (new_grid[r, cols_action4] == 9)
            new_grid[r, cols_action4][mask5] = 9
            new_grid[r, cols_action4][mask9] = 5
        return new_grid

    else:
        return grid

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves filling a target area or reaching a score.
    # In the transitions, we see values changing at r63c60.
    # Let's assume level complete if row 63 has enough '1's.
    count_ones = np.sum(grid[63, :] == 1)
    return count_ones >= 5