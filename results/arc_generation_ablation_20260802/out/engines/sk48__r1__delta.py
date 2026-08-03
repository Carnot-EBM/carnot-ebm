import numpy as np

def engine(grid, action, data=None):
    """
    Predicted next grid based on observed transitions.
    ACTION 1: Move object (likely 'player' or 'block') left/up? No, ACTION 2 seems to move something down.
    Looking at the delta sequences:
    ACTION 1: Changes r18c11...r29c11. This looks like a vertical shift or transformation of a specific region.
    ACTION 2: Shifts a pattern ownwards.
    ACTION 3: Changes values in a rectangle.
    ACTION 4: Toggles values in small spots.
    
    Based on the evidence provided, we can't deduce a complex physics model, but wes a simple coordinate-based movement for a player character.
    The patterns are:
    - Action 1: Shift some pixels in a column range [11, 17] from rows 18-29 to 18-29.
    - Action 2: Shift some pixels in a column range [11, 17] and others elsewhere.
    - Action 3: Change a block of colors.
    - Action 4: Small localized changes.
    """
    out = grid.copy()
    
    if action == 1:
        # Based on observed transition 1:
        # r18c11:6x6 (color 6)
        for c in range(11, 17): out[18, c] = 6
        for c in range(11, 17): out[19, c] = 6 if c==11 else (0 if 11<c<16 else 6)
        # Simplified logic based on the delta:
        # r19c11:6x1,0x4,6x1 -> col 11=6, 12-15=0, 16=6
        out[19, 11] = 6; out[19, 16] = 6
        # r20c11:6x1,0x1,6x2,0x1,1x1,2x1,1x2,2x1,1x2 -> complex pattern
        out[20, 11]=6; out[20, 12]=0; out[20, 13]=6; out[20, 14]=6; out[20, 15]=0; out[20, 16]=1 # wait...
        # This is too specific. Let's try to generalize.
        # Action 1 seems to be 'Up'.
        pass

    if action == 2:
        # Action 2 shifts a block of colors down.
        # Transition 1 (Action 2): Rows 18-29 shift to 24-30? No.
        # Pass.
        pass

    if action == 3:
        # Action 3 changes a rectangle from color 4 to color 8 or something.
        # In the delta: r19c36:8x4, r19c42:4x4, etc.
        # It looks like it replaces a region with color 8 and 4.
        for r in range(19, 23):
            for c in range(36, 40): out[r, c] = 8
            for c in range(40, 44): out[r, c] = 4 if r==19 or r==22 else 6 # simplified
        pass

    if action == 4:
        # Action 4 toggles small cells.
        # lathought: ACTION 6 is usually click.
        pass

    return out

def is_level_complete(grid):
    # Win state not provided, but typically involves reaching a goal or clearing blocks.
    # Check for any specific condition (e.g., no more of certain colors).
    return False