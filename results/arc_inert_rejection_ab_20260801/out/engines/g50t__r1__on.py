import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the player moves a "cursor" or "block" 
    # and changes colors of specific regions.
    # Based on the observed transitions, ACTION2 (Right/Down?) shifts patterns of color change.
    # Action 2 is likely 'Move Right'.
    # Action 4 is likely 'Move Down'.
    # Action 6 is click.
    
    # However, the provided deltas are very complex and involve changing large blocks of cells.
    # In order to implement this world model, we umare looking for thes rules:
    #
    # 1. Color 5 is often the background/wall.
    # 2. Colors 2, 8, 9 are interactable objects.
    # 3. There's a//a pattern of movement that progresses through the same relative coordinates.
    # 4. The grid contains some "slots" or "targets".
    # 5. The laout suggests a maze-like structure with a target at r63c60+.
    
    # Let's try to induce a simple rule based on the transition data.
    # Transition 1: ACTION2 shifted something from c14 to c20.
    # # Looking closer at the deltas:
    # # r8c14:5x5, r8c20:2x5... then r8c20:5x5, r8c26:2x5...
    # # This looks like a block of size 5x5 (color 5) and a block of size 5x5 (color 2) moving right by 6 columns.
    # # This is essentially shifting a window of modification.
    
    # We can see that the blocks being changed are actually replacing existing colors.
    # # Action 2 shifts this "active area" active_col = current_col + 6.
    # # Action 4 shifts this "active area" active_row = current_row + 6? No, let's<|channel>thought//t
    # # Actually, looking at the delta for ACTION4: r50c14:5x5, r50c20:9x5... it changes cells in rows 50-54.
    # # The previous action was ACTION2 which had modified rows 44-48.
    # # So ACTION4 moves the focus to the next set of rows.
    
    # Since we don't have the state stored between calls to engine(), we must infer the cursor position from the grid itself.
    # Let's look for a unique marker. In the INITIAL GRID, there are some color 2s and 9s.
    # Color 1 is only at r63c62.
    # In Transition 1, r63c61 becomes 1. Then r63c60, then r63c59, etc.
    # This means the "player" or "cursor" is moving leftwards on row 63 as they perform actions.
    
    # We can actually see that the player is essentially "painting" or "clearing" blocks.
    # Each Action 2 (Right) shifts the active block right by 6 columns.
    # Each Action 4 (Down) shifts the active block down.
    
    # But wait, the most reliable indicator of progress is the cell with value 1 on row 63.
    # The cells r63c62, r63c61, r63c60... are being flipped to 1 one by one.
    # Let's track the leftmost '1' on row 63.
    
    # For ACTION2:
    # - It moves the cursor in row 63 one step to the left.
    # - It modifies a specific set of blocks based on current position.
    # - Looking at the deltas, it seems to be replacing color 2/8/9 with 5 and vice versa.
    
    # However, without knowing the exact mapping of which action affects which coordinate, 
    # we should implement the simplest possible logic that matches the observed transitions.
    
    # Given the constraints and the nature of ARC-AGI world models, if the grid state itself 
    # doesn't explicitly store the "cursor", we must assume the engine is called in sequence 
    # and the grid reflects the state.
    
    # Let's look for the marker '1' on row 63.
    marker_col = np.where(grid[63] == 1)[0]
    if len(marker_col) > 0:
        current_pos = np.min(marker_col)
    else:
        current_pos = 62

    next_grid = grid.copy()
    
    if action == 2: # Move Right / Progress
        # The cursor moves left on row 63
        if current_pos > 0:
            next_grid[63, current_pos - 1] = 1
        
        # Now apply a block change based on current_pos.
        # This part is very specific to the level layout.
        # We can see ACTION2 modifies blocks at c=14, 20, 26, 32... (increments of 6).
        # The offset seems to be related to how many times Action 2 was pressed.
        times_pressed = 62 - current_pos
        offset = times_pressed * 6
        
        # Based on deltas:
        # Press 1: r8-12 c14->5, c20->2; r14-18 c14->9
        # Press 2: r8-12 c20->5, c26->2; r20-24 c14->9
        # Press 3: r8-12 c26->5, c32->2; r26-30 c14->9
        # ... and so on.
        
        # Let's implement this pattern:
        # Top block shift:
        next_grid[8:13, 14+offset : 19+offset] = 5
        next_grid[8:13, 20+offset : 25+offset] = 2
        
        # Side block progression:
        side_row_start = 14 + (times_pressed // 1) * 6 # This is a guess
        # Looking at the data: press 1 -> row 14, press 2 -> row 20, press 3 -> row 26...
        side_row_start = 14 + times_pressed * 6
        if side_row_start < 64:
            next_grid[side_row_start : side_row_start+5, 14:19] = 9
            # And then it seems to be cleared by subsequent presses?
            # Press 2 clears rows 14-18 and sets 20-24 to 9.
            # So we clear previous side blocks:
            prev_side_row = side_row_start - 6
            if prev_side_row >= 14:
                next_grid[prev_side_row : prev_side_row+5, 14:19] = 5

    elif action == 4: # Move Down / Special
        # ACTION4 changes r50-54 c14->5, c20->9
        next_grid[50:55, 14:19] = 5
        next_grid[50:55, 20:25] = 9
        # It doesn't seem to move the marker on row 63.

    return next_grid

def is_level_complete(grid):
    # Level complete when the marker reaches a certain point or grid matches win state.
    # In this case, let's assume it's complete when the marker reaches column 0.
    return grid[63, 0] == 1

def is_level_complete(grid):
    # A level is complete if the grid contains no longer
    # the same pattern of the same color.
    # This is a a simplified general rule for the same-color-//-
    # same-pattern pattern.
    return True if grid.any() else False
