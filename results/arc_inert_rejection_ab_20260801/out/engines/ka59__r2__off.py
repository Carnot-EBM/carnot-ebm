import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Inducing rules from observed transitions:
    # Action 2: Up? No, based on delta, it seems to be a movement or shift of a pattern.
    # Action 3: Right/Left? Based on deltas, r33c18 -> r33c15. This looks like shifting left.
    # Action 4: Down? Or some other operation.
    # Action 2: r30-32 c21 shifted to r33-35 c21. Shift down.
    # Action 3: r33-35 c18 shifted to r33-35 c15. Shift left.
    # Action 4: r30-32 c18 shifted to r30-32 c21. Shift right.
    # Let's refine this:
    # The object being moved is a 3x3 block of color 1 and others.
    # The "player" is likely the entity moving through the maze.
    # Action 2: Move Down
    # Action 3: Move Left
    # Action 4: Move Right
    # Action 1: Move Up (implied)
    # Action 2: Down, Action 3: Left, Action 4: Right, Action 1: Up.
    
    # Find the current position of the player (the unique pattern).
    # The player is characterized by the 3x3 block starting at (r, c).
    # The player is often color 1.
    # The<|channel>thoughtt process suggests the player is a 3x3 area.
    # However, looking at the deltas, it's simpler:
    # ACTION 4 moves something from c18 to c21 (+3 cols)
    # ACTION 2 moves something from r30 to r33 (+3 rows)
    # ACTION 3 moves something from c21 to c18 (-3 cols), then c18 to c15 (-3 cols)
    # This looks like movement in increments of 3.
    
    # Let's identify the "player" as the 3x3 region that changes.
    # We need to find where the '1's are concentrated.
    # Since we don't have the full grid state transitions, let's use the delta logic.
    
    # In this specific game, the object moving is likely the set of cells with value 1.
    # But wait, the INITIAL GRID shows many 1s.
    # Let's look for the most compact group of 1s or a specific pattern.
    # Actually, the simplest model is: Action 1=Up, 2=Down, 3=Left, 4=Right.
    # Shift by 3 units.
    
    # To implement this, we first need to locate the player.
    # The player seems to be a 3x3 block of color 1 (or containing 1).
    # Looking at the deltas: r30c18:1x3... becomes r30c21:1x3...
    # It means the values at c18-20 were replaced and values at c21-23 were updated.
    
    # Let's assume the "player" is the 3x3 block of color 1 that is NOT part of the walls (color 2/15).
    # We will find all 3x3 blocks of color 1 and pick one? No, let's just track the movement.
    # Since we don't have a clear starting point, we can search for the 3x3 block of '1's.
    
    # Correcting action mapping based on observed transitions:
    # ACTION 4: Right (+3 cols)
    # ACTION 2: Down (+3 rows)
    # ACTION 3: Left (-3 cols)
    # ACTION 1: Up (-3 rows)
    
    # Find the current position of the 3x3 block of 1s.
    # In the initial grid, there are several 1s. The player is likely the one that moves.
    # We look for a 3x3 area where cells are mostly 1.
    
    # For this specific task, since the board is large and patterns are sparse,
    # we can identify the player by finding any 3x3 block of 1s that isn't "static".
    # But in engine(), we only have the current grid.
    # Let's assume the player is the 3x3 block of 1s located at (r, c).
    
    # To find the player:
    # Search for a 3x3 block of color 1.
    # If multiple exist, we need a way to distinguish.
    # However, looking at the INITIAL GRID, there's a clear structure.
    # r21-25: c9-23 is 1... wait.
    # Let's just implement the movement logic.
    
    new_grid = grid.copy()
    
    # Locate the 'player' - a 3x3 block of 1s.
    # Since it's hard to uniquely identify, let's search for the most likely candidate.
    # The observed transitions happen around r30, c18-21.
    # Let's look for a 3x3 block of 1s near those coordinates or generally.
    
    player_pos = None
    for r in range(64 - 2):
        for c in range(64 - 2):
            if np.all(grid[r:r+3, c:c+3] == 1):
                player_pos = (r, c)
                break
        if player_pos: break

    if player_pos is None:
        return new_grid

    r, c = player_pos
    dr, dc = 0, 0
    if action == 1: dr = -3
    elif action == 2: dr = 3
    elif action == 3: dc = -3
    elif action == 4: dc = 3
    
    nr, nc = r + dr, c + dc
    
    if 0 <= nr < 64-2 and 0 <= nc < 64-2:
        # Move the 3x3 block: clear old, set new.
        # But wait, the deltas show that it replaces whatever was there.
        # And the "walls" are color 2 and 15.
        # Let's check if the target position is blocked by walls.
        blocked = False
        for i in range(3):
            for j in range(3):
                if grid[nr+i, nc+j] in [2, 15]:
                    blocked = True
                    break
            if blocked: break
        
        if not blocked:
            # Clear current pos (set to background? The delta shows it becomes something else)
            # In the observed transitions, the cells at the old position are replaced by 14 or other values.
            # This suggests a swap or a specific fill.
            # Looking at ACTION 4: r30c18:1x3... became 14x3.
            # So the player leaves behind color 14.
            new_grid[r:r+3, c:c+3] = 14
            new_grid[nr:nr+3, nc:nc+3] = 1
            
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a goal or clearing objects.
    # No win state provided, but typically it's when the player reaches a certain area.
    # Based on the deltas, there's a cell r63c63 that changes to 0.
    # This might be a countdown or a target.
    # Let's assume completion if a specific condition is met.
    # Since we don't have the win grid, let's return False unless a known win condition is found.
    return False