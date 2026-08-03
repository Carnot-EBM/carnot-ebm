import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # The object being moved is a 3x3 block of color 9 (or similar).
    # It moves in steps of 6 columns/rows.
    # The target cell at the end of the same row (r63) is updated to 0.
    # Based on thes deltas, it seems like a puzzle where a block moves and leaves a trail or replaces values.
    # Let's implement a movement logic for the 3x3 block of color 9.
    
    new_grid = grid.copy()
    H, W = new_grid.shape
    
    # Find current position of the 3x3 block of color 9
    # We look for any cell that color 9 exists.
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    # Get the top-left corner of the 3x3 block
    r, c = coords[0]
    # Since we are part of a larger structure, therethought be multiple blocks of 9s.
    # But only one "active" block moving.
    # The active block is not just 3x3, but potentially other colors.
    # The laest transition shows r21c33:0x3, r27c33:9x3 etc.
    # This means at (21, 33), the cells became 0, and it moved to (27, 33).
    # # ACTION2: Down (row +6)
    # ACTION3: Left (col -6)
    # ACTION4: Right (col +6)
    
    # Identify the active block' same size as the 3x3 area.
    # Find all coordinates of color 9.
    # Let's assume the active block is 3x3.
    # find the center or top-left.
    #<|channel>thought
    # In the first delta: ACTION2 moves from r21 to r27. Row diff = 6.
    # ACTION3 moves from c33 to c27. Col diff = -6.
    # ACTION4 moves from c21 to c27 then c27 to c33. Col diff = +6.
    
    # To implement this correctly, we a need to identify which 3x3 block of 9s is moving.
    # The one that changes in the deltas.
    # We look for the most recently changed region.
    # But engine() must be pure based on grid state.
    # We look for the "active" block. In these examples, it's only one block of 9s.
    # 
    # Let's refine: find the 3x3 block of 9s.
    # If there are multiple, we might need more info. But here, only one.
    # 
    # Find all cells with color 9.
    # Get the bounding box of the active block.
    # Since other colors (like 0 or 5) also change, let's just track the 9s.
    # 
    # Movement logic:
    # ACTION2: r += 6
    # ACTION3: c -= 6
    # ACTION4: c += 6
    
    # Find current position of the 3x3 block of 9s
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    # Assume the moving object is a 3x3 block of 9s.
    # Group coordinates into blocks.
    # Find the one that is not aligned to some static grid? No, they are aligned.
    # The most likely candidate for "active" is the one that can move.
    # In these examples, it starts at r21c33 and moves.
    # Let's find any 3x3 block of 9s.
    # For simplicity, since only one moves in the trace:
    r_min, c_min = coords[0] # This might be wrong if there are multiple blocks.
    # But we need to know which one is active.
    # Looking at INITIAL GRID, color 9 is at r21-23, c33-35 (approx).
    # Wait, let's check INITIAL GRID again.
    # r21: ... 9x3 ... (at col 33?)
    # Summing counts for r21: 21+3+3+3+3=33. Yes, r21c33 is 9x3.
    # So initial position is r=21, c=33.
    
    # To handle movement correctly, we must clear the old position and set the new one.
    # What happens to the cells?
    # ACTION2: r21c33 becomes 0x3, r27c33 becomes 9x3.
    # It seems it replaces whatever was there with 9s, and leaves 0s behind?
    # No, look at r27c33 in INITIAL: it was 0x3. Now it's 9x3.
    # And r21c33 was 9x3. Now it's 0x3.
    # This looks like a simple "move" of a 3x3 block of color 9.
    # The value left behind is 0.
    
    # Let's find all 3x3 blocks of 9s.
    # Since only one moves, let's just track any 3x3 block of 9s that can move.
    # In this specific level, there's only one moving block.
    
    # Find top-left of the active 3x3 block of 9s.
    # We search for a 3x3 area of 9s.
    for r in range(H - 2):
        for c in range(W - 2):
            if np.all(grid[r:r+3, c:c+3] == 9):
                curr_r, curr_c = r, c
                break
        else: continue
        break
    else: return new_grid

    dr, dc = 0, 0
    if action == 2: dr = 6
    elif action == 3: dc = -6
    elif action == 4: dc = 6
    else: return new_grid
    
    new_r, new_c = curr_r + dr, curr_c + dc
    
    if 0 <= new_r < H - 2 and 0 <= new_c < W - 2:
        # Clear old position to 0
        new_grid[curr_r:curr_r+3, curr_c:curr_c+3] = 0
        # Set new position to 9
        new_grid[new_r:new_r+3, new_c:new_c+3] = 9
        
        # Also update the "progress bar" at r63
        # The deltas show r63c55:0x1, then r63c54:0x1...
        # This means as it moves, a cell in row 63 is set to 0.
        # Let's find which cell in r63 is currently non-zero and set it to 0.
        # Or just follow the pattern: each move reduces the index of the last non-zero cell?
        # Actually, let's look at the delta for ACTION2 (first one): r63c55:0x1.
        # Then ACTION2 again: r63c54:0x1.
        # It seems every action sets one more cell in r63 to 0, starting from c55 downwards.
        # Let's implement this simply: find the rightmost non-zero cell in r63 and set it to 0.
        # Wait, INITIAL GRID r63: 6x56, 0x8. So cells 0-55 are color 6, 56-63 are color 0.
        # Delta 1: r63c55 becomes 0. Now 55-63 are 0.
        # Delta 2: r63c54 becomes 0. Now 54-63 are 0.
        # This means we set grid[63, current_rightmost_color_6] = 0.
        
        last_col = -1
        for col in range(W - 1, -1, -1):
            if new_grid[63, col] == 6:
                last_col = col
                break
        if last_col != -1:
            new_grid[63, last_col] = 0

    return new_grid

def is_level_complete(grid):
    # The level is complete when the block reaches a target or row 63 is all 0s?
    # In these examples, it's not completed yet.
    # Usually, completion is when the moving object reaches a specific goal.
    # Let's assume it's complete if the block of 9s reaches some position or r63 is empty.
    # Given no win state, let's guess based on common ARC patterns.
    # Maybe when the block of 9s hits color 14 (the same as the "goal" blocks)?
    coords = np.argwhere(grid == 9)
    if coords.size == 0: return False
    r_min, c_min = coords[0]
    # Check if any part of the 3x3 block overlaps with color 14.
    for r in range(r_min, r_min + 3):
        for c in range(c_min, c_min + 3):
            if r < grid.shape[0] and c < grid.shape[1] and grid[r, c] == 14:
                return True
    return False