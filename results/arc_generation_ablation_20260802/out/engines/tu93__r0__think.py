import numpy as np

def engine(grid, action, data):
    """
    The game 'tu93' involves moving a 3x3 block of color 9 across the grid.
    Based on the observed transitions:
    ACTION2: Move Down
    ACTION3: Move Left
    ACTION4: Move Right
    
    The block moves between specific slots defined by the layout.
    Looking at the deltas:
    Initial position was around r21-23, c33 (implied by first ACTION2 delta).
    Wait, looking closer at the INITIAL GRID and DELTAS:
    r21-23, c33 had value 9 in some rows? No, let's re-examine.
    
    Actually, the pattern shows that when an action is taken, a 3x3 area of color 9
    is cleared from its current location and placed into a new 3x3 area.
    Additionally, there is a "progress bar" or counter at r63c50+ that decrements/changes.
    
    Let's track the 3x3 block of 9s:
    Start: The initial grid doesn't explicitly show where the 9s are in a simple way, 
    but the first ACTION2 delta says r21c33:0x3, r22c33:0x3, r23c33:0x3 become 0.
    This means the block of 9s was at (21, 33) to (23, 35).
    Then it moves to r27c33, r28c33, r29c33 becoming 9.
    So ACTION2 moves the block Down by 6 units.
    ACTION3 moves the block Left by 6 units.
    ACTION4 moves the block Right by 6 units.
    """
    new_grid = grid.copy()
    
    # Find current position of the 3x3 block of 9s
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    # Assume the block is a 3x3 square; find top-left corner
    r_min, c_min = coords.min(axis=0)
    
    # Clear current block
    new_grid[r_min:r_min+3, c_min:c_min+3] = 0 # Or whatever background color is appropriate
    # Looking at deltas, they are replaced by 0 or other values. 
    # But usually in these games, moving means clearing old and setting new.
    # The delta says "r21c33:0x3", meaning it becomes color 0.
    
    dr, dc = 0, 0
    if action == 2: # Down
        dr = 6
    elif action == 3: # Left
        dc = -6
    elif action == 4: # Right
        dc = 6
    else:
        return new_grid

    nr, nc = r_min + dr, c_min + dc
    
    # Boundary check (logical grid is 64x64)
    if 0 <= nr < 61 and 0 <= nc < 61:
        new_grid[nr:nr+3, nc:nc+3] = 9
        
    # Handle the progress bar at r63
    # The observed transitions show cells at r63 changing to 0 from right to left.
    # This seems like a side effect of movement.
    # We can't perfectly induce the logic for r63 without more data, but we can try to mimic.
    # However, usually the core puzzle is the block movement.
    
    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the block reaches a target or the counter is empty.
    Given the limited data, we look for the block reaching a specific area 
    or the bottom-right corner of the board becoming color 0.
    """
    # In many ARC games, completion is defined by the final state of the object.
    # Based on the provided delta, it moves towards the bottom/left/right.
    # Let's check if the block has reached the same region as the '14' colors (target).
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return False
    r_min, c_min = coords.min(axis=0)
    # Target area was around r45-47, c33-35
    if r_min >= 45 and c_min >= 33:
        return True
    return False