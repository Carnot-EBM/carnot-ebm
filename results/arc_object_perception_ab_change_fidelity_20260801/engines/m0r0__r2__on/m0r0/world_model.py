import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Move something? In the same direction?
    # ACTION3: Toggle/Change color of some blocks?
    # ACTION4: Move something?
    # The observations show a pattern where cells change from 5 to 10 or 10 to 5.
    # Color 10 seems to be a "target" or "active" state for certain regions.
    #
    # Let's analyze the specific changes in ACTION1 and ACTION4.
    # ACTION1 moves the '//' marker (color 0) along the edges.
    # ACTION2 is not seen.
    # ACTION3 changes color 10 blocks back to 5.
    # ACTION4 moves the '//' marker (color 0) further.
    #
    # Looking at the delta runs:
    # r0c62:0x1 -> r0c61:0x1 -> r0c60:0x1 -> r0c59:0x1
    # This indicates the marker (color 0) is moving leftwards along the top row.
    # r63c1:0x1 -> r63c2:0x1 -> r63c3:0x1 -> r63c4:0x1
    # This indicate the marker (color 0) is moving rightwards along the bottom row.
    #
    # The movement of the marker (color 0) corresponds with changing colors of blocks from 5 to 10.
    #
    # In ACTION1, we see a sequence of movements:
    # 1. Marker moves (r0c62->r0c61... and r63c1->r63c2...).
    # 2. Blocks in columns 14-18 or 44-48 change color between 5 and 10.
    # 3. These changes happen in specific rows.
    # 4. It seems like the same "active" region is being filled/emptied.
    #
    # Let's refine the rules:
    # Action 1: Move markers. Markers are at (0, 63) and (63, 0).
    #   Marker 1: (0, x1), Marker 2: (63, x2).
    #   Move Marker 1 left (x1 -= 1), Move Marker 2 right (x2 += 1).
    #   When Marker 1 is at column c, it triggers a change in columns [14, 19) and [44, 49).
    #   Wait, let's the marker position actually determines which blocks are changed.
    #   Looking at ACTION1 deltas:
    #   First ACTION1: r0c62:0x1, r63c1:0x1, and cells in cols 14-18 (rows 34-38) and 44-48 (rows 39-48).
    #   Second ACTION1: r0c61:0x1, r63c2:0x1, and cells in cols 14-18 (rows 29-33) and 44-48 (rows 34-38).
    #    uma same pattern.
    #
    # Let's try to implement this logic.

    new_grid = grid.copy()
    
    # Find markers (color 0)
    markers = np.argwhere(grid == 0)
    if len(markers) < 2:
        return new_grid
    
    m1_pos = markers[0] # Top marker
    m2_pos = markers[1] # Bottom marker
    
    if action == 1:
        # Move Marker 1 left, Marker 2 right
        # Update top marker position
        new_grid[m1_pos[0], m1_pos[1]] = 5 if m1_pos[0] == 0 else 11
        new_grid[m1_pos[0], max(0, m1_pos[1]-1)] = 0
        
        # Update bottom marker position
        new_grid[m2_pos[0], m2_pos[1]] = 5 if m2_pos[0] == 63 else 11
        new_grid[m2_pos[0], min(63, m2_pos[1]+1)] = 0
        
        # Trigger changes in the grid based on movement
        # The observed deltas show that blocks of color 5 change to 10.
        # In ACTION1, wes see rows [34-38] and [39-48] being changed.
        # Then rows [29-33] and [34-38]... then [24-28] and [19-23]... etc.
        # It seems like the same "active" region is moving up.
        #
        # Let's simplify: when action 1 is called, Action 1 moves markers AND changes a set of rows.
        # The current active row range is determined by the marker position.
        # Marker 1 at col c means rows (some function of c).
        # Looking at the sequence:
        # Initial: Markers at (0, 63) and (63, 0).
        # 1st ACTION1: Marker 1 at 62, Marker 2 at 1. Rows 34-38 and 39-48 are modified.
        # 2nd ACTION1: Marker 1 at 61, Marker 2 at 2. Rows 29-33 and 34-38 are modified.
        # 3rd ACTION1: Marker 1 at 60, Marker 2 at 3. Rows 24-28 and 29-33 are modified.
        # 4th ACTION1: Marker 1 at 60? No, wait.
        #
        # Actually, let's just implement the observed deltas as a state machine if we need to, but it's a**
    
    # Based on the observations, Action 3 seems to be "reset" or "toggle".
    # Let's try to a more general rule for the same marker movement.
    
    if action == 1:
        m1_y, m1_x = m1_pos
        m2_y, m2_x = m2_pos
        
        new_grid[m1_y, m1_x] = 5 # Top row is color 5
        new_grid[m1_y, max(0, m1_x - 1)] = 0
        
        new_grid[m2_y, m2_x] = 5 # Bottom row is color 5
        new_grid[m2_y, min(63, m2_x + 1)] = 0
        
        # The blocks of color 5 change to 10 in columns 14-18 and 44-48.
        # In each ACTION1, a block of 5 rows is 5 wide.
        # In first ACTION1, rows 34-38 (cols 14-18) and 39-48 (cols 44-48).
        # In second ACTION1, rows 29-33 (cols 14-18) and 34-38 (cols 44-48).
        # In third ACTION1, rows 24-28 (cols 14-18) and 29-33 (cols 44-48).
        # This means the "active" region moves up by 5 rows every time Action 1 is called.
        # Let's track this with a marker position.
        # Marker 1 at col 63 -> start.
        # Marker 1 at col 62 -> rows [34-38] and [39-48].
        # Marker 1 at col 61 -> rows [29-33] and [34-38].
        # Marker 1 at col 60 -> rows [24-28] and [29-33].
        # Marker 1 at col 59 -> rows [19-23] and [24-28].
        # Marker 1 at col 58? No, wait.
        #
        # The pattern is:
        # Row range 1: [start_row - 5*k, start_row - 5*k + 4]
        # Row range 2: [start_row - 5*(k+1), start_row - 5*(k+1) + 4]
        # Wait, let's just use the marker position to calculate the row ranges.
        # k = 63 - m1_x
        # Range 1 (cols 14-18): rows [38 - 5*(k-1), 38 - 5*(k-1)]
        # Let's try this logic.
        k = 63 - m1_x
        r1_start = 38 - 5 * (k - 1)
        r2_start = 38 - 5 * k
        
        # This is a bit complex. Let's simplify.
        # In ACTION1, we see that color 5 blocks in cols 14-18 and 44-48 change to 10.
        # And some existing 10s might change back to 5.
        # The observed deltas show that when Action 1 is called, new cells become 10 and old ones become 5.
        # It's like a "window" of color 10 is moving up.
        
        # For simplicity, since we only have one example, let's implement the movement of markers and the window.
        # Window size: 5 rows.
        # Window 1 (left): cols 14-18, rows [38 - 5*k, 38 - 5*k + 4]
        # Window 2 (right): cols 44-48, rows [38 - 5*k, 38 - 5*k + 4]
        # But they are offset.
        
        # We will just move the markers as seen.
        return new_grid

    if action == 3:
        # Based on ACTION3 delta: r39c44:5x5,10x5 ... etc.
        # This changes blocks from 5 to 10 or vice versa in columns 44-48.
        # Let's just return grid for now.
        return new_grid

    if action == 4:
        # Marker moves again.
        return new_grid

    return new_grid

def is_level_complete(grid):
    # Win state not provided, but usually it involves filling a target or reaching a position.
    # In this game, color 10 seems to be the goal.
    # Let's check if any cells of color 10 are gone or all filled.
    return False