import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game involves a "cursor" or "player" represented by color 1 at (63, 62) in initial state.
    # Action 2 is likely 'Right' and Action 4 is 'Left'.
    # There are blocks/objects that move or change colors based on player movement.
    # Based on observed transitions, ACTION2 moves the cursor (color 1) leftwards along row 63.
    # ACTION4 moves the cursor (color 1) leftwards? No, let's re-examine.
    #
    # Transition analysis:
    # Initial: r63c62=1.
    # After first ACTION2: r63c61=1.
    # After second ACTION2: no mention of r63C... but wait, it says r63c60=1 in third transition.
    # Let's look at the deltas again.
    # r63c61:1x1 -> r63c60:1x1 -> r63c59:1x1 -> r63c58:1x1.
    # So ACTION2 moves the cursor (color 1) from col 62 to 61, then 60, 59, 58.
    # This is moving LEFT.
    #
    # Now look at the "changed cells" for ACTION2 and ACTION4.
    # ACTION2 changes blocks of color 5/9/8/2.
    # The grid contains a large structure of color 5 (obj6).
    # There_is_a_pattern of shifting colors within that structure.
    # It seems like a "sliding puzzle" or "shifting block" game.
    # When the player (color 1) moves left, certain regions of the grid are updated.
    # Specifically, the region between columns 14-38 (approx) is being modified.
    #
    # Looking at the ACTION2 transitions:
    # Transition 1: r8c14:5x5, r8c20:2x5... r14c14:9x5...
    # Transition 2: r8c20:5x5, r8c20:2x5... r20c14:9x5...
    # Transition 3: r8c26:5x5, r8c26:2x5... r26c14:9x5...
    # Transition 4: r8c32:5x5, r8c32:2x5... r32c14:9x5...
    # This looks like a sequence of shifts. Each ACTION2 move pushes a "window" of changes.
    #
    # The cursor movement is simple: find color 1 and move it one cell to the left if possible.
    #
    # For the complex block changes: they seem to be tied to the player's position.
    # Let's look at the cursor positions: (63, 62), (63, 61), (63, 60), (63, 59), (63, 58).
    # These are associated with specific grid updates.
    #
    # Given the constraints and the nature of ARC-AGI, we can actually deduce that this is likely a "mirroring" or "shifting" mechanism.
    # However, since the only observed actions are ACTION2 and ACTION4, and let's assume ACTION2=Left, ACTION4=Right.
    # But wait, ACTION4 also moves blocks?
    # ACTION4 delta: r50c14:5x5, r50c20:9x5...
    #
    # Actually, looking closely at the deltas, the "changed cells" for ACTION2 are very structured.
    # They are replacing sections of the same size (5x5) with other colors.
    # This looks like it's moving a pattern.
    #
    # Let's implement the simple cursor movement and the block changes as they appear in the data.
    # Since we cannot induce a general rule for the complex patterns without more data, but we umare limited to the laout.
    # We will focus on the cursor movement and the logic that seems most consistent.

    new_grid = grid.copy()
    
    # Find player position
    player_pos = np.where(grid == 1)
    if len(player_pos[0]) == 0:
        return new_grid
    py, px = player_pos[0][0], player_pos[1][0]
    
    if action == 2: # Move Left
        px -= 1
        if px < 0: px = 0
    elif action == 4: # Move Right
        px += 1
        if px >= new_grid.shape[1]: px = new_grid.shape[1] - 1
    
    # Update player position
    new_grid[py, px] = 1
    # The old position is now color 9 (based on r63c61:1x1 etc.)
    # Wait, looking at the deltas: "r63c61:1x1" means cell (63, 61) becomes 1.
    # It doesn't explicitly say what happens to the previous cell.
    # But in a typical ARC game, it would be replaced by the background color or the original color of that cell.
    # In row 63, the initial grid is 9x62, 1x2. So cells 0-61 are 9 and 62-63 are 1? No, 9x62 means 62 cells of value 9. Then 1x2 means 2 cells of value 1.
    # So r63 has 9s from col 0 to 61, and 1s from col 62 to 63.
    # When ACTION2 moves cursor to 61, r63c61 becomes 1.
    # Let's assume the player replaces the cell they move into and the cell they leave becomes 9.
    
    # We need to find where the player was.
    old_py, old_px = py, px + (1 if action == 2 else -1 if action == 4 else 0)
    if 0 <= old_px < new_grid.shape[1]:
        new_grid[old_py, old_px] = 9
        
    return new_grid

def is_level_complete(grid):
    # Level complete when the player reaches a certain position or collects something.
    # In this game, it's not shown, but usually it's reaching the left side.
    player_pos = np.where(grid == 1)
    if len(player_pos[0]) == 0:
        return False
    py, px = player_pos[0][0], player_pos[1][0]
    return px == 0