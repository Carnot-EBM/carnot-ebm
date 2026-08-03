import numpy as np

def engine(grid, action, data=None):
    if action != 6:
        return grid
    
    y, x = data['y'], data['x']
    out = grid.copy()
    
    # The game seems to be about clicking on "buttons" (color 0) or "targets" (color 14, 15, 9, 11).
    # Based on the observed transitions, ACTION6 at specific coordinates triggers changes.
    # Let's analyze the same-coordinate clicks.
    # Click (36, 59) -> r56c33:0x6, etc. This looks like clearing a region.
    # Click (23, 30) -> r28c21:9x4, etc. This looks like filling a target area.
    # Click (20, 59) -> r56c17:0x6, etc.
    # Click (29, 30) -> r28c27:14x4, {r53c62:3x1}, etc.
    # Click (44, 59) -> r56c41:0x6, etc.
    # Click (35, 30) -> r35c33:11x4, etc.
    
    # Mapping click coords to effects based on evidence:
    # (36, 59) -> clears color 0 in a 6x6 block starting at (56, 33)? No, it's r56c33:0x6... wait, the delta is "changed cells".
    # If the cell was already 0, and it becomes 0, it wouldn't be in the delta.
    # The same coordinates are clicked twice for each pair of buttons/targets.
    # Clicking (36, 59) first makes some cells 0. Then clicking (23, 30) fills them with 9.
    # Let's look at the<|channel>thought
    # Target areas:
    # Area 1: x=21-24, y=28-31 (color 9)
    # Area 2: x=27-30, y=28-31 (color 14)
    # Area 3: x=33-36, y=28-31 (color 11)
    # Area 4: x=17-20, y=56-61? No.
    
    # Looking at the deltas again:
    # Click (36, 59): r56c33:0x6, r57c33:0x1... this looks like a "hole" being punched in color 4.
    # Click (23, 30): r28c21:9x4, etc. This fills target area 1 with color 9.
    # It also changes r56c33 to 4.
    # So Clicking (36, 59) opens a hole, and clicking (23, 30) closes it and fills a target.
    
    # Let's refine the coordinates:
    # Hole A: top-left (56, 33), size 6x6 (but some cells are not mentioned).
    # Hole B: top-left (56, 17), size 6x6.
    # Hole C: top-left (56, 41), size 6x6.
    
    # Target Area 1: (28, 21) to (31, 24), color 9.
    # Target Area 2: (28, 27) to (31, 30), color 14.
    # Target Area 3: (28, 33) to (31, 36), color 11.
    
    # Trigger Coords:
    # Button A: (59, 36) -> Opens Hole A.
    # Button B: (59, 20) -> Opens Hole B.
    # Button C: (59, 44) -> Opens Hole C.
    # Fill A: (30, 23) -> Fills Target 1 with 9 and closes Hole A.
    # Fill B: (30, 29) -> Fills Target 2 with 14 and closes Hole B.
    # Fill C: (30, 35) -> Fills Target 3 with 11 and closes Hole C.
    
    # Let's implement this logic.
    if x == 36 and y == 59:
        out[56, 33:39] = 0
        for r in range(57, 61): out[r, 33] = 0; out[r, 38] = 0
        out[61, 33:39] = 0
    elif x == 20 and y == 59:
        out[56, 17:23] = 0
        for r in range(57, 61): out[r, 17] = 0; out[r, 22] = 0
        out[61, 17:23] = 0
    elif x == 44 and y == 59:
        out[56, 41:47] = 0
        for r in range(57, 61): out[r, 41] = 0; out[r, 46] = 0
        out[61, 41:47] = 0
    elif x == 23 and y == 30:
        out[28:32, 21:25] = 9
        out[56, 33:39] = 4
        out[61, 33:39] = 4
        for r in range(57, 61): out[r, 33] = 4; out[r, 38] = 4
        # Special case from evidence: r53c63:3x1. This is likely a score/progress marker.
        out[53, 63] = 3
    elif x == 29 and y == 30:
        out[28:32, 27:31] = 14
        out[56, 17:23] = 4
        out[61, 17:23] = 4
        for r in range(57, 61): out[r, 17] = 4; out[r, 22] = 4
        out[53, 62] = 3
    elif x == 35 and y == 30:
        out[28:32, 33:37] = 11
        out[56, 41:47] = 4
        out[61, 41:47] = 4
        for r in range(57, 61): out[r, 41] = 4; out[r, 46] = 4
        out[53, 61] = 3

    return out

def is_level_complete(grid):
    # Level complete when all target areas are filled.
    # Target Area 1: (28, 21) to (31, 24), color 9.
    # Target Area 2: (28, 27) to (31, 30), color 14.
    # Target Area 3: (28, 33) to (31, 36), color 11.
    if np.all(grid[28:32, 21:25] == 9) and \
       np.all(grid[28:32, 27:31] == 14) and \
       np.all(grid[28:32, 33:37] == 11):
        return True
    return False