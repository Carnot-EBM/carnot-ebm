import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    if action != 6:
        return grid.copy()
    
    y, x = data['y'], data['x']
    
    # The game seems to be a puzzle where clicking on certain areas triggers changes.
    # Based on the observed transitions, there are specific "trigger" zones and "target" zones.
    # Trigger zones are at y=59 (bottom area) and y=30 (middle area).
    # Let's analyze the same-color blocks that move or change.
    # There are four distinct colors involved in movement/change: 9, 14, 11, 15.
    # These are the same colors found in both top and bottom target areas.
    
    # Define the color mapping for the trigger points based on observations.
    # Mapping from click coordinates (x, y) to the color it activates.
    # # Action 6 (x=36, y=59) -> r56c33:0x6, etc. This looks like clearing a block.
    # # Action 6 (x=23, y=30) -> r28c21:9x4, etc. Color 9 moves to middle.
    # # Action 6 (x=20, y=59) -> r56c17:0x6, etc. Clearing a block.
    # # Action 6 (x=20, y=59) is followed by (x=29, y=30) which moves color 14.
    # # laction 6 (x=44, y=59) -> r56c41:0x6, {etc}
    # # Action 6 (x=20, y=59) clears the block at x=17-22 range.
    # # Action 6 (<|channel>thought
    # The game involves clicking buttons (blocks of colors 9, 14, 11, 15).
    # There are blocks in the top area (y=1..6), bottom area (y=57..60), and middle area (y=28..31).
    # Clicking on a button in the bottom area seems to "unlock" or "clear" something.
    # Clicking on a button in the middle area seems to move a colored block from the top to the middle.
    # Wait, looking closer at the deltas:
    # (x=23, y=30) triggers Color 9 moving to (r28-31, c21-24).
    # (x=29, y=30) triggers Color 14 moving to (r28-31, c27-30).
    # (x=35, y=30) triggers Color 11 moving to (r28-31, c33-36).
    # And clicks at y=59 clear the corresponding color blocks at the bottom.
    
    # Let's refine the logic based on observed transitions.
    # The colors are 9, 14, 11, 15.
    # Bottom buttons are located around x = [17-21], [26-30], [33-37], [41-45].
    # Middle trigger points are around x = [23], [29], [35], [41?].
    
    new_grid = grid.copy()
    
    # Trigger coordinates and associated colors/actions
    # Format: (click_x, click_y), target_color, middle_pos_x, bottom_pos_x
    triggers = [
        {'cx': 23, 'cy': 30, 'color': 9,  'mx': 21, 'bx': 17},
        {'cx': 29, 'cy': 30, 'color': 14, 'mx': 27, 'bx': 17}, # Wait, bx is different in observations
        {'cx': 35, 'cy': 30, 'color': 11, 'mx': 33, 'bx': 41},
    ]
    
    # Re-evaluating based on the actual deltas provided:
    # Action (x=36, y=59) -> clears r56c33 to r61c38 (Color 9 block at bottom?)
    # Action (x=23, y=30) -> Color 9 moves to r28-31, c21-24. Also modifies r53c63 and r56-61, c33-38.
    # Action (x=20, y=59) -> clears r56c17 to r61c22.
    # Action (x=29, y=30) -> Color 14 moves to r28-31, c27-30. Modifies r53c62 and r56-61, c17-22.
    # Action (x=44, y=59) -> clears r56c41 to r61c46.
    # Action (x=35, y=30) -> Color 11 moves to r28-31, c33-36. Modifies r53c61 and r56-61, c41-46.
    
    # It seems:
    # Click middle x=23 -> color 9 moves to mid, bottom block at x=33 is replaced by grid background (color 4).
    # Click middle x=29 -> color 14 moves to mid, bottom block at x=17 is replaced by grid background (color 4).
    # Click middle x=35 -> color 11 moves to mid, bottom block at x=41 is replaced by grid background (color 4).
    # Click bottom x=36, y=59 -> clears bottom block at x=33.
    # Click bottom x=20, y=59 -> clears bottom block at x=17.
    # Click bottom x=44, y=59 -> clears bottom block at x=41.

    if y == 59:
        if 17 <= x <= 22: # Bottom button 1
            new_grid[56:62, 17:23] = 4
        elif 33 <= x <= 38: # Bottom button 2
            new_grid[56:62, 33:39] = 4
        elif 41 <= x <= 46: # Bottom button 3
            new_grid[56:62, 41:47] = 4
    elif y == 30:
        if 21 <= x <= 25: # Mid trigger 1
            new_grid[28:32, 21:25] = 9
            new_grid[53, 63] = 3
            new_grid[56:62, 33:39] = 4
        elif 27 <= x <= 31: # Mid trigger 2
            new_grid[28:32, 27:31] = 14
            new_grid[53, 62] = 3
            new_grid[56:62, 17:23] = 4
        elif 33 <= x <= 37: # Mid trigger 3
            new_grid[28:32, 33:37] = 11
            new_grid[53, 61] = 3
            new_grid[56:62, 41:47] = 4

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually it involves moving all blocks.
    # Based on the deltas, maybe filling r53 with color 3? Or clearing bottom buttons.
    # Let's check if any of the target colors are in the middle area and bottom buttons are gone.
    # For now, return False as no WIN STATE grid was provided to induce from.
    return False