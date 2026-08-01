import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    # Action 2 is a right movement or shift in some context.
    # Action 4 is a left movement or shift in some context.
    # Action 6 is a click.
    
    # Based on observed transitions, ACTION2 seems to move a "block" or "window" of color changes.
    # The pattern shows that when ACTION2 is called, certain regions change from color 0/8/9 to 5 and vice versa.
    # Let's identify the "active" area (the large region of color 5).
    # In the initial grid, there's a large mass of color 5 starting at r7c13.
    # # Looking at the deltas for ACTION2:
    # # Transition 1: r8c14:5x5, r8c20:2x5... (color 5 replaces something, then color 2 replaces it)
    # # lathought: This looks like a sliding window.
    # # A block of size 5x5 (or similar) moves across the board.
    # # The cells are changed to color 5, then color 2.
    # # Also, vertical movements occur.
    # # There's a<|channel>thought: The object with shape 'f4a5824013a7fc6f' (obj6) is the main body.
    # # It seems we are filling in holes (color 0 or 8) within the boundaries of obj6.
    # # The "brush" is moving.
    # # Action 2 shifts the brush right.
    # # Action 4 shifts the brush left.
    # # Action 6 would be a click.
    
    # Let's refine the rules based on the observed transitions:
    # ACTION2: Moves the current "active area" to the right by 6 columns.
    # ACTION4: Moves the current "active area" to the left by 6 columns.
    # In each step, the new area becomes color 5 and the old area becomes something else.
    # But wait, the deltas show that as it moves right, it also descends vertically?
    # r8c14 -> r8c20 -> r8c26 -> r8c32...
    # And then later: r20c14 -> r26c14 -> r32c14 -> r38c14...
    # This looks like a sequence of operations.
    
    # Actually, looking at the deltas again:
    # Transition 1: (r8-12, c14-19) becomes 5, (r8-12, c20-24) becomes 2. (r14-18, c14-18) becomes 9.
    # Transition 2: (r8-12, c20-24) becomes 5, (r8-12, c26-30) becomes 2. (r20-24, c14-18) becomes 9.
    # Transition 3: (r8-12, c26-30) becomes 5, (r8-12, c32-36) becomes 2. (r26-30, c14-18) becomes 9.
    # Transition 4: (r8-12, c32-36) becomes 5, (r8-12, c38-42) becomes 2. (r32-37, c14-18) becomes 9...
    # This is a "painting" process. The brush moves right across the top row of holes, then jumps down and repeats.
    # And it seems to be filling color 0/8 with color 5 or 9.
    
    # Let's look at the win condition. In the INITIAL grid, r63 has color 9x62 and 1x2.
    # After ACTION2 transitions, we see r63c61:1x1, r63c60:1x1, etc.
    # Color 1 is moving leftward in the bottom row.
    # This suggests that as we "paint" the rest of the board, the goal is to move the block of color 1 to the far left.
    
    new_grid = grid.copy()
    if action == 2:
        # Move brush right.
        # We need to find where the "brush" currently is.
        # Looking at the deltas, the brush size is roughly 5x5.
        # It fills cells with 5, and leaves a trail of 2 (or something else).
        # But wait, the deltas are very specific about coordinates.
        # The most consistent thing is the movement of color 1 on r63.
        # Each ACTION2 moves the pixel of color 1 one step to the left.
        for r in range(64):
            for c in range(64):
                if r == 63 and new_grid[r, c] == 1:
                    if c > 0:
                        new_grid[r, c] = 9
                        new_grid[r, c-1] = 1
    elif action == 4:
        # Move brush left.
        # Based on Action 2 moving it left, Action 4 might move it right?
        # Or maybe Action 4 does something else. In the observed data, only one Action 4 occurs.
        # It changes r50c14:5x5, r50c20:9x5...
        # This looks like it's filling a hole with 5 and another with 9.
        pass

    # To implement this perfectly, we need to track the "brush" position.
    # But since we don't have the brush start pos, let's use the most reliable signal: color 1 movement.
    return new_grid

def is_level_complete(grid):
    # The level is complete when the pixel of color 1 reaches the far left (column 0).
    # Check if grid[63, 0] == 1.
    return grid[63, 0] == 1

import numpy as np

def is_level_complete(grid):
    # The win state is when all cells in the grid are the same color (all zeros).
    # This is a a general rule for many ARC-AGI games.
    return np.all(grid == 0)
