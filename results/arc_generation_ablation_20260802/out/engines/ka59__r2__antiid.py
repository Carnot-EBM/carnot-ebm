import numpy as np

def engine(grid, action, data):
    # The game seems to involve moving blocks of color 14 (and potentially others)
    # based on keyboard actions. ACTION2=Up, ACTION3=Right, ACTION4=Left, ACTION5=Down?
    # Or some other mapping. Based on the delta changes:
    # ACTION4 (left) moves things leftward or modifies them.
    # ACTION3 (right) moves things rightward.
    # ACTION2 (up) moves things up.
    # ACTION5 (down) would move things down.
    
    new_grid = grid.copy()
    
    if action == 6: # Click
        return new_grid
    
    # Identify "active" objects - contiguous regions of non-background colors (excluding background 2 and 4)
    # Let's assume the same active object is the laest modified region or a player character.
    # In this case, it's likely that the blocks of color 14 are being shifted.
    # Wethought Action 4 is Left, 3 is Right, 2 is Up, 5 is Down.
    
    # Looking at the observed transitions:
    # ACTION4: r30c18... shifts something to the left.
    #// Wait, looking closer at the deltas:
    # // ACTION4 (level 0->0): changed cells r30c18:1x3,14x3 ...
    # // This looks like a block of size 3x3 was moved or replaced.
    # // It seems there are multiple blocks of color 14.
    # // The logic might be moving a specific cursor or a set of blocks.
    
    # Simple heuristic based on the observations:
    # ACTION4 (Left): Shift all color 14 blocks left by 3 units?
    # ACTION3 (Right): Shift all color 14 blocks right by 3 units?
    # ACTION2 (Up): Shift all color 14 blocks up by some amount?
    
    # Let's try a move-all approach for these actions.
    # For simplicity and let's use the same mapping:
    # Action 2: Up, 3: Right, 4: Left, 5: Down.
    
    if action == 4: # Left
        # Find all 14s, shift them left.
        mask = (grid == 14)
        new_grid[mask] = 2 # Clear old positions
        # Shift mask left by 3 columns
        shifted_mask = np.roll(mask, -3, axis=1)
        new_grid[shifted_mask] = 14
        return new_grid

    if action == 3: # Right
        mask = (grid == 14)
        new_grid[mask] = 2
        shifted_mask = np.roll(mask, 3, axis=1)
        new_grid[new_grid == 0] = 2 # Clean up any zeros that might be replaced
        new_grid[shifted_mask] = 14
        return new_grid

    if action == 2: # Up
        mask = (grid == 14)
        new_grid[mask] = 2
        shifted_mask = np.roll(mask, -3, axis=0)
        new_grid[shifted_mask] = 14
        return new_grid
    
    if action == 5: # Down
        # Not seen in data but logically follows the pattern.
        mask = (grid == 14)
        # Clear old positions
        new_grid[mask] = 2
        # { a block of 14s was moved from r30-32 to r33-35 }
        # Shift mask down by 3 rows.
        shifted_mask = np.roll(mask, 3, axis=0)
        new_grid[shifted_mask] = 14
        return new_grid

    return new_grid

def is_level_complete(grid):
    # Usually complete when certain blocks reach a target or are gone.
    # Since no win state grid was provided, we assume it's not complete yet.
    # return True if any color 14 reaches the same area as color 15?
    # No clear goal. Let's just check for something that other than background.
    return False