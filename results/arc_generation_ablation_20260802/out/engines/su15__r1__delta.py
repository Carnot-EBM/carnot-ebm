import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid
    
    px, py = data['x'], data['y']
    out = grid.copy()
    
    # The observed transitions show ACTION6 (click) at (px, py) creating a 3x3 block of color 15
    # and replacing another 3x3 block of color 15 elsewhere.
    # Based on thes patterns:
    # (10, 53) -> r52c9:15x3, r53c9:15x3, r54c9:15x3 AND r58c3:5x3, r59c3:5x3, r60c3:5x3
    # (16, 47) -> r46c15:15x3, r47c15:15x3, r48c15:15x3 AND r52c9:5x3, r53c9:5x1, 3x1, 5x1...
    # It seems clicking creates a 3x3 square of color 15 centered at (py-1, px-1) or similar.
    # Let's refine coordinates:
    # Click (10, 53) -> Block at row 52, 53, 54; col 9, 10, 11. Center is (53, 10).
    # Click (16, 47) -> Block at row 46, 47, 48; col 15, 16, 17. Center is (47, 16).
    # Click (22, 41) -> Block at row 40, 41, 42; col 21, 22, 23. Center is (41, 22).
    # Click (28, 35) -> Block at row 34, 35, 36; col 27, 28, 29. Center is (41, 28)? No, center is (35, 28).
    # Click (34, 29) -> Block at row 28, 29, 30; col 33, 34, 35. Center is (29, 34).
    # In all cases: New block center = (py, px). Rows: py-1 to py+1, Cols: px-1 to px+1.
    
    # The "replaced" block is the one that was previously created by a click or the initial 3x3 color 15 blocks.
    # Initial blocks of color 15 are at r4,5,6 c30,31,32 and r58,59,60 c3,4,5.
    # Initial grid has color 15 blocks at (4,30), (5,30), (6,30) etc. and (58,3), (59,3), (60,3).
    # Let's find existing 3x3 blocks of color 15.
    # We only care about the same ones as observed in the transitions.
    # The logic seems to be: move the 3x3 square of color 15 from its current position to the new clicked position.
    
    # Find current positions of all 3x3 blocks of color 15.
    # Since we only track one moving block based on evidence, let's assume there is one "active" block.
    # In transition 1: Click (10, 53) -> Block moves from (58,3) to (53,10).
    # In transition 2: Click (16, 47) -> Block moves from (53,10) to (47,16).
    # In transition 3: Click (22, 41) -> Block moves from (40, 41)? No, center is (41, 22).
    # Wait, look at the deltas again.
    # Transition 1: r52c9:15x3, r53c9:15x3, r54c9:15x3 AND r58c3:5x3, r59c3:5x3, r60c3:5x3.
    # This means a 3x3 square of 15s was at (58,3), (59,3), (60,3) and became 5s.
    # The new 3x3 square of 15s is at (52,9), (53,9), (54,9).
    # Center of old block: (59, 4). Center of new block: (53, 10).
    # Let's check Transition 2: New block at (46, 15), (47, 15), (48, 15). Old block at (52, 9), (53, 9), (54, 9).
    # Center of old block: (53, 10). Center of new block: (47, 16).
    # Logic: Move the most recently created 3x3 block of color 15 to the clicked position.
    
    # Find the "active" 3x3 block of color 15.
    # We can find it by scanning for any 3x3 block of color 15.
    # In the initial grid, there are two blocks: one at r4-6 c30-32 and one at r58-60 c3-5.
    # The evidence shows only the same one moving.
    # The one at r4-6 c30-32 is never mentioned in deltas.
    # The one at r58-60 c3-5 is the first one moved.
    # So we move the block that is NOT at (4,30) etc.
    
    # Identify all 3x3 blocks of color 15.
    blocks = []
    for r in range(grid.shape[0] - 2):
        for c in range(grid.shape[1] - 2):
            if np.all(grid[r:r+3, c:c+3] == 15):
                blocks.append((r, c))
            # To avoid overlapping detections, we skip a few cells if<|channel>thought
            # But here they are exactly 3x3.
    
    # We need to find which one to "erase".
    # The evidence suggests the one most recently moved or the one starting at (58,3).
    # If multiple exist, and one is at (4,30), ignore it.
    # target_block = None
    # for b in blocks:
    #     if b != (4, 30):
    #         target_block = b
    #     elif len(blocks) == 1: # only one exists
    #         # this might be the same block
    #         pass
    
    # Find any 3x3 block of color 15 that isn't the static one at (4,30).
    active_block = None
    for r, c in blocks:
        if not (r == 4 and c == 30):
            active_block = (r, c)
            break
    
    if active_block is None:
        # Fallback: use the first one found.
        if blocks:
            active_block = blocks[0]
    
    if active_block:
        # Erase old block: set to background color (color 5 based on deltas).
        # The deltas show "r52c9:5x3" etc., which means they become color 5.
        # Let's check if we can determine background color from grid.
        bg_color = 5
        out[active_block[0]:active_block[0]+3, active_block[1]:active_block[1]+3] = bg_color
        
    # Create new block at clicked position.
    # Click (px, py) -> Block rows [py-1, py+1], cols [px-1, px+1].
    # Wait, let's re-verify coordinates again.
    # Click (10, 53) -> r52, 53, 54; c9, 10, 11.  (py=53, px=10)
    # Rows are 53-1, 53, 53+1. Cols are 10-1, 10, 10+1. Correct.
    new_r, new_c = py - 1, px - 1
    out[new_r : new_r + 3, new_c : new_c + 3] = 15
    
    return out

def is_level_complete(grid):
    # No win state provided in evidence, but typically it involves moving the block to a target.
    # The deltas show "r63c62:5x2", "r63c60:5x2" etc. which looks like a progress bar or counter.
    # Let's assume complete if the active block reaches a certain position.
    # But without a WIN STATE grid, we can only guess.
    # Usually, ARC levels are complete when a specific pattern is achieved.
    # Given the data, let's return False unless we see a clear goal.
    return False