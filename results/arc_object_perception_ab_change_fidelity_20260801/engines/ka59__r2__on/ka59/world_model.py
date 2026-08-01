import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    # Action mapping based on observed transitions:
    # ACTION 2: Up? No, looking at deltas, it seems like a movement of an object.
    # Let's analyze the movements from the INITIAL GRID and the same-object tracking.
    # The object being moved is likely the small 3x3 blocks or similar.
    # ACTION 4: Right/Down?
    # ACTION 3: Left?
    # ACTION 2: Down?
    # Based on the delta r30c18...r32c18 (obj6), obj6 moves from (30,18) to (30,21)?
    # ACTION 4: Move right.
    # ACTION 3: Move left.
    # ACTION 2: Move down.
    # ACTION 1: Move up.
    # ACTION 6: Click.
    
    # We need to find the "active" object. In this game, there are often specific objects that can be moved.
    # The active object is usually the one most recently interacted with or a certain color.
    # Since we only have one set of transitions, let's assume the active object is the 3x3 block of color 14.
    # Action 4: Right, Action 3: Left, Action 2: Down, Action 1: Up.
    
    # Find all objects of color 14.
    # Let's identify the same-object tracking.
    # obj6 and obj7 are color 14 blocks.
    # Let's laught at the logic: it's a simple movement of a 3x3 block of color 14.
    # a = np.where(np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]]) == 1) # shape id 8063231412b5aae6
    # This is a<|channel>thought
    # a = np.where(np.array([[1, 1, 1], [1, 0, 1], [ uma own loop]) == 1)
    # { "ACTION 4": "Right", "ACTION 3": "Left", "ACTION 2": "Down", "ACTION 1": "Up" }
    
    # In this game, there_is an active object. The most likely candidate for the "active" object
    # is the one that moves.
    # Looking at the deltas:
    # ACTION 4 (level 0->0): r30c18:1x3,14x3 -> obj6 moved from c18 to c21.
    # ACTION 4 again: r30c21:1x3,14x3... -> obj6 moved from c21 to c24? No, wait.
    # Let's look closer: r30c18:1x3,14x3 means cells starting at col 18 are now color 1 and then color 14.
    # So the block of color 14 was at c18-20 and it's now at c21-23.
    # This confirms Action 4 = Right, Action 3 = Left, Action 2 = Down, Action 1 = Up.
    
    # Now we need to determine which object is moving.
    # There are two blocks of color 14. Which one moves?
    # In these games, usually there's a state variable or the "last clicked" object.
    # Since no clicks were observed before movement, maybe the first one found?
    # Or maybe only one can move?
    # Looking at the deltas:
    # ACTION 4 (level 0->0): changed cells r30c18:1x3,14x3 ... r32c18:1x3,14x3.
    # The original obj6 was at bbox=(30,18,32,20). After this action, it's at (30,21,32,23).
    # Then ACTION 4 again: r30c21:1x3,14x3... -> moved from c21 to c24? No, wait.
    # Let's re-read: "r30c18:1x3,14x3". This means col 18,19,20 become 1 and col 21,22,23 become 14.
    # Yes, that is moving Right by 3 pixels.
    # Action 4 = Right (+3), Action 3 = Left (-3), Action 2 = Down (+3), Action 1 = Up (-3).
    
    # Which object moves? In the observed transitions, only one block of color 14 moves.
    # It starts at (30,18) and then later another block of color 14 (obj7) might move?
    # Actually, looking at the deltas, only one block seems to be active.
    # Let's assume the first block of color 14 found in a row-major scan is the active one.
    
    # To implement this, we need to track the position of the active object.
    # Since the engine must be pure, we can't have external state.
    # We must identify the "active" object from the grid itself.
    # Maybe it's the one not aligned with some grid? Or maybe there's only one that *can* move?
    # Looking at the initial grid: obj6 is at (30,18), obj7 is at (30,27).
    # The movements are all for the same object.
    # Let's try moving the first block of color 14 encountered.
    
    # Wait, look at ACTION 4 again: r30c26:14x1,1x3... This looks like obj7 moved!
    # So both blocks of color 14 can move? No, they probably move based on some rule.
    # In many ARC games, Action 4/3/2/1 moves ALL objects of a certain type or the "selected" one.
    # But here, different actions seem to move different things.
    # Actually, let's re-examine:
    # Transition 1: ACTION 4 -> obj6 moves Right.
    # Transition 2: ACTION 4 -> obj6 moves Right again.
    # Transition 3: ACTION 4 -> obj7 moves Right.
    # Transition 4: ACTION 3 -> obj6 moves Left.
    # Transition 5: ACTION 2 -> obj6 moves Down.
    # Transition 6: ACTION 3 -> obj6 moves Left.
    # Transition 7: ACTION 3 -> obj6 moves Left.
    # It seems only one object is active at a time. How is it selected?
    # Maybe clicking (ACTION 6) selects it? But no clicks are shown.
    # Wait, maybe the action itself determines which one moves if there's an ambiguity?
    # Or maybe I should just move all blocks of color 14 that *can* move in that direction?
    # Let's try moving any block of color 14 that has space.
    
    # Let's refine the movement:
    # The blocks are 3x3 "rings" (color 14 on border, something else inside).
    # They move by 3 units.
    
    dy = {1: -3, 2: 3, 3: 0, 4: 0}
    dx = {1: 0, 2: 0, 3: -3, 4: 3}
    
    if action not in dy:
        return grid
    
    new_grid = grid.copy()
    
    # Find all 3x3 rings of color 14
    for r in range(64):
        for c in range(64):
            if grid[r, c] == 14:
                # Check if it's the top-left of a 3x3 ring
                if r + 2 < 64 and c + 2 < 64:
                    is_ring = True
                    for dr in range(3):
                        for dc in range(3):
                            if dr == 1 and dc == 1: continue
                            if grid[r+dr, c+dc] != 14:
                                is_ring = False; break
                        if not is_ring: break
                    
                    if is_ring:
                        # Try to move this ring
                        nr, nc = r + dy[action], c + dx[action]
                        if 0 <= nr and nr + 2 < 64 and 0 <= nc and nc + 2 < 64:
                            # Move it: clear old, set new
                            # This is tricky because we might overwrite other things.
                            # The observed deltas show the same cells becoming color 1 (the background).
                            # Let's assume the "background" for these rings is color 1.
                            for dr in range(3):
                                for dc in range(3):
                                    if dr == 1 and dc == 1: continue
                                    new_grid[r+dr, c+dc] = 1
                                    new_grid[nr+dr, nc+dc] = 14
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when objects reach a target.
    # In this case, maybe when they are at certain positions?
    # Since no win state was given, return False.
    return False