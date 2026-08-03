import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where some object (color 9 or 4) moves across the same-colored blocks.
    # Based on the observed transitions, it looks like color 9/4 acts as a "cursor" or "player" that moves in cardinal directions.
    # Let's identify the current position of the player (the cell containing 9 or 4).
    # In the initial grid, there are multiple instances of <|channel>thought
    # However, looking at the deltas, only one specific set of 3x3 blocks move.
    # We look for the 'active' block which is color 9 or 4.
    # For simplicity, we will track the movement of the 3x3 block of color 9.
    # The active block is always 3x3.
    # Action 2: Down
    # Action 3: Left
    # Action 4: Right
    # Action 5: Up (not seen but implied by symmetry)
    # Action 1: Not seen
    # Action 6: Click
    # Action 7: Not seen
    # Action 8: Not seen
    # Action 0: No action
    #
    # Looking at the delta: r21c33:0x3 r22c33:0x3 r23c33:0x3 r27c33:9x3 r28c33:9x3 r29c33:9x1,4x1,9x1
    # This means a 3x3 block moved from (21-23, 33-35) to (27-29, 33-35).
    # It moves in jumps of 6 rows.
    #
    # Let's refine the player position. The "player" is the 3x3 block that changes color to 9.
    # Find current 3x3 block of color 9.
    # Find its top-left corner (r, c).
    # Move it based on the action.
    #
    # Action 2: Down -> r += 6
    # Action 3: Left  -> c -= 6
    # Action 4: Right -> c += 6
    # Action 5: Up    -> r -= 6
    #
    # When moving, the old position becomes color 0 or 5?
    # In the delta: r21c33:0x3 ... r27c33:9x3. Old cells become 0. New cells become 9.
    # Also there is some change at r63c... this looks like a progress bar.

    # Find the active 3x3 block of color 9 (or 4)
    # We search for the first instance of color 9 or 4.
    coords = np.argwhere(np.isin(grid, [9, 4]))
    if len(coords) == 0:
        return grid
    
    # Assume the player is the most recently changed block, but since we don't have state,
    # let's just find the top-leftmost coordinate of the cluster.
    r_min, c_min = coords.min(axis=0)
    
    # Move logic
    dr, dc = 0, 0
    if action == 2: # Down
        dr = 6
    elif action == 3: # Left
        dc = -6
    elif action == 4: # Right
        dc = 6
    elif action == 5: # Up
        dr = -6
    else:
        return grid
        
    new_r, new_c = r_min + dr, c_min + dc
    
    # Update grid
    new_grid = grid.copy()
    # Clear old position (set to 0 as seen in deltas)
    for i in range(3):
        for j in range(3):
            if 0 <= r_min+i < 64 and 0 <= c_min+j < 64:
                new_grid[r_min+i, c_min+j] = 0
                
    # Set new position to color 9
    for i in range(3):
        for j in range(3):
            if 0 <= new_r+i < 64 and 0 <= new_c+j < 64:
                # The delta shows some cells might be color 4, but mostly 9.
                # We'll use 9 for simplicity unless we see a specific pattern.
                new_grid[new_r+i, new_c+j] = 9

    # Progress bar logic at row 63
    # Find the current "empty" part of the progress bar (color 0).
    # It seems to be filling from right to left or vice versa.
    # In ACTION2: r63c55:0x1 -> r63c54:0x1 -> r63c52:0x2...
    # This means the number of zeros is increasing? No, it says "changed cells".
    # If r63c55 was something and now is 0, then 0s are appearing.
    # Let's just simulate the movement and ignore the progress bar if not critical for win state.
    
    return new_grid

def is_level_complete(grid):
    # Win state is usually when a certain condition is met.
    # Since no win state grid was provided, we assume complete when player reaches a target.
    # Target might be color 14 blocks.
    coords_player = np.argwhere(np.isin(grid, [9, 4]))
    coords_target = np.argwhere(grid == 14)
    if len(coords_player) == 0 or len(coords_target) == 0:
        return False
    
    # Check if any part of the player block overlaps with target area (color 14)
    for p in coords_player:
        for t in coords_target:
            if np.linalg.norm(p - t, ord=1) <= 2: # Close enough to overlap
                return True
    return False