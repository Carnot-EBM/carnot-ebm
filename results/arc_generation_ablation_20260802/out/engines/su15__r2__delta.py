import numpy as np

import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    out = grid.copy()
    
    # The observed transitions show a pattern where clicking on certain coordinates (x, y)
    # creates a 3x3 square of color 15 at (py, px) and replaces another 3x3 square of color 15
    # with color 5.
    # In each case, the click coordinate corresponds to the center of a new 3x3 square.
    # Note that the<|channel>thought process suggests the same x, y are clicked.
    # Let's identify the target area for the 3x3 squares.
    # r4c30:15x3, r5c30:15x3, r6c30:15x3 is the initial state.
    # Transition 1: Click (10, 53) -> New 3x3 at r52-54, c9-11; Old 3x3 at r58-60, c3-5 becomes 5.
    # Transition 2: Click (16, 47) -> New 3x3 at r46-48, c15-17; Old 3x3 at r52-54, c9-11 becomes 5.
    # Transition 3: Click (22, 41) -> New 3x3 at r40-42, c21-23; Old 3x3 at r40-42, c21-23 was not there yet?
    # Wait, let's re-examine the deltas.
    # T1: click(10, 53) -> new square at r52..54, c9..11; old square at r58..60, c3..5 becomes 5.
    # T2: click(16, 47) -> new square at r46..48, c15..17; old square at r52..54, c9..11 becomes 5.
    # T3: click(22, 41) -> new square at r40..42, c21..23; old square at r46..48, c15..17 becomes 5.
    # T4: click(28, 35) -> new square at r28..30, c27..29? No, r34..36, c27..29; old square at r40..42, c21..23 becomes 5.
    # T5: click(34, 29) -> new square at r28..30, c33..35; old square at r34..36, c27..29 becomes 5.
    
    # The pattern is: the clicked point (px, py) is the center of a 3x3 block.
    # New block: out[py-1:py+2, px-1:px+2] = 15
    # Old block: find where color 15 was and change it to 5.
    # But wait, there's also changes in row 63.
    # Let's look at the coordinates again.
    # Click (10, 53): py=53, px=10. Block centered at (53, 10). Range [52:55, 9:12].
    # Click (16, 47): py=47, px=16. Block centered at (47, 16). Range [46:49, 15:18].
    # Click (22, 41): py=41, px=22. Block centered at (41, 22). Range [40:43, 21:24].
    # Click (28, 35): py=35, px=28. Block centered at (35, 28). Range [34:37, 27:30].
    # Click (34, 29): py=29, px=34. Block centered at (29, 34). Range [28:31, 33:36].
    
    # Now find the "old" block that becomes color 5.
    # T1: r58-60, c3-5 became 5.
    # T2: r52-54, c9-11 became 5.
    # T3: r46-48, c15-17 became 5.
    # T4: r40-42, c21-23 became 5.
    # T5: r34-36, c27-29 became 5.
    
    # The old block is always the one created by the previous click.
    # For T1, it was a pre-existing block at r58-60, c3-5? No, initial grid has blocks at r4-6, c30-32 and r58-60, c3-5.
    # Let's check INITIAL GRID for r58-60, c3-5: r58:5x3, 15x3... Yes, there is a 15x3 block starting at col 3.
    
    # So the rule is:
    # 1. Create new 3x3 of color 15 centered at (py, px).
    # 2. Find all existing 3x3 blocks of color 15 and change them to color 5.
    # Wait, only ONE block changes to 5? In T1, only r58-60, c3-5 changed.
    # But in T1, the original block at r4-6, c30-32 did NOT change.
    # Maybe it's the "most recent" or "closest" block? Or a specific sequence?
    # Looking at the coordinates: (53,10) -> (47,16) -> (41,22) -> (35,28) -> (29,34).
    # These are moving up and right. The block being replaced is also moving up and right.
    # It seems the block that was created by the previous action is the one replaced.
    # For the first action, the same logic applies if we assume there was a "previous" block at (59, 4).
    
    # Let's refine:
    # New square at [py-1:py+2, px-1:px+2] = 15.
    # Find any other 3x3 square of 15s and turn it into 5s? No, just the most recent one.
    # Since we don't have state for "most recent", let's look for ANY 3x3 square of 15s that isn't the new one.
    # But wait, T1 only replaces r58-60, c3-5. What about r4-6, c30-32? That stays 15.
    # Maybe it replaces the 3x3 block closest to the click? Or the one with the highest y?
    # In T1: Click(53,10), Block(59,4) is closer than Block(5,31).
    # In T2: Click(47,16), Block(53,11) is closer than Block(5,31).
    # This seems consistent.
    
    # Also row 63 changes:
    # T1: r63c62:5x2 (cols 62, 63 become 5)
    # T2: r63c60:5x2 (cols 60, 61 become 5)
    # T3: r63c58:5x2 (cols 58, 59 become 5)
    # T4: r63c56:5x2 (cols 56, 57 become 5)
    # T5: r63c54:5x2 (cols 54, 55 become 5)
    # The columns are 62, 60, 58, 56, 54... decreasing by 2 each time.
    # This looks like a progress bar or counter.
    
    # Let's implement the block movement and the row 63 change.
    # To handle "most recent", we can search for any 3x3 of color 15 that is NOT at the new position.
    # But to be safe and match T1, let's find the one closest to the click.
    
    for r in range(py-1, py+2):
        for c in range(px-1, px+2):
            out[r, c] = 15
            
    # Find existing 3x3 blocks of 15s
    blocks = []
    for r in range(grid.shape[0]-2):
        for c in range(grid.shape[1]-2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                blocks.append((r, c))
    
    if blocks:
        # Find the block closest to (py, px)
        best_block = min(blocks, key=lambda b: abs(b[0]+1 - py) + abs(b[1]+1 - px))
        br, bc = best_block
        for r in range(br, br+3):
            for c in range(bc, bc+3):
                out[r, c] = 5

    # Row 63 change: This is tricky because it depends on how many actions have occurred.
    # Since engine() is pure and doesn't know action count, we can infer it from row 63.
    # If cols 62,63 are 0, they become 5. If 60,61 are 0...
    # Let's check current state of row 63.
    for col in range(64-2, -1, -2):
        if grid[63, col] == 0 and grid[63, col+1] == 0:
            out[63, col] = 5
            out[63, col+1] = 5
            break
            
    return out

def is_level_complete(grid):
    # No win state provided, but usually it's when a goal is reached.
    # Based on the progress bar, maybe when row 63 is filled?
    return np.all(grid[63, :] == 5)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state for game 'su15'.
    The win state is typically a sorted sequence of numbers (1 to N-1) 
    followed by a blank space (0) in a sliding puzzle.
    """
    grid = np.array(grid)
    flat = grid.flatten()
    
    # For a standard 15-puzzle (4x4), the win state is 1, 2, ..., 15, 0
    # Total elements = 16
    n = flat.size
    expected = np.arange(1, n)
    expected = np.append(expected, 0)
    
    return np.array_equal(flat, expected)
