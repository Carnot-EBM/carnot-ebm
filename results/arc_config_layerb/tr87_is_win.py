import numpy as np

def is_win(grid):
    # The EDITABLE region is rows 48..63, cols 15..63.
    # Looking at the grid structure:
    # Rows 0-31 contain patterns of 0s, 5s, 7s, and 2s.
    # Rows 32-47 contain patterns of 3s, 0s, 5s, 7s, and 1s.
    # Row 63 is all 1s.
    # The "rule" or "reference" is often a pattern shown in a non-editable area.
    # In this specific configuration (tr87), the pattern in rows 4-10 and 22-28 
    # (the 0/5/7 patterns) and rows 50-56 (the 3/0/5/7 patterns) are key.
    
    # Specifically, the pattern in rows 50-56 (the 3/0/5/7 pattern) 
    # is the target for the EDITABLE region (rows 48-63).
    # However, looking at the grid, the EDITABLE region (48-63, 15-63) 
    # is currently filled with 3s.
    # The pattern at rows 50-56, cols 15-63 is:
    # Row 50: 3333333333333333333333333333333333333333333333333333333333333333
    # Row 51: 3333333333333333333333333333333333333333333333333333333333333333
    # Row 52: 3333333333333333333333333333333333333333333333333333333333333333
    # Row 53: 3333333333333333333333333333333333333333333333333333333333333333
    # Row 54: 3333333333333333333333333333333333333333333333333333333333333333
    # Row 55: 3333333333333333333333333333333333333333333333333333333333333333
    # Row 56: 3333333333333333333333333333333333333333333333333333333333333333
    
    # Wait, the pattern at rows 50-56 is actually the "reference" for the 
    # EDITABLE region (48-63).
    # Let's re-examine the grid. 
    # Rows 50-56, cols 15-63 are all 3s.
    # The EDITABLE region is rows 48-63, cols 15-63.
    # In many tr87 puzzles, the EDITABLE region must match a specific 
    # sub-pattern shown in the "rule" area.
    # The rule area is the block at rows 50-56, cols 15-63.
    
    editable_region = grid[48:64, 15:64]
    reference_region = grid[50:57, 15:64]
    
    # In this specific puzzle, the rule is that the editable region 
    # (48-63) must match the reference region (50-56) shifted or 
    # simply match the pattern of the 3s/0s/5s/7s block.
    # Looking at the grid, the 3s block is at 50-56.
    # The editable region is 48-63.
    # The most common win condition is that the editable region 
    # matches the reference block exactly in its relative position.
    
    # Let's check the 3s block at 50-56.
    # If we align the editable region (48-63) such that its 
    # rows 50-56 match the reference 50-56:
    
    # Actually, the most likely rule is that the editable region 
    # (48-63) must be all 3s, as shown in the reference block.
    
    return np.all(editable_region == 3)