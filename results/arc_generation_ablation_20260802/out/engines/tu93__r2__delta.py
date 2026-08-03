import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the same rules induced from observed transitions.
    """
    # Grid dimensions are 64x64.
    # Observed actions: ACTION2 (down), ACTION3 (left), ACTION4 (right).
    # There's a 'player' object (color 9) which moves within a 'maze' of walls (color 5) and paths (colors 0, 2, 14).
    # The player is color 9.
    # Color 5 is wall/background.
    # Color 0 is path.
    # Color 2 is obstacle/wall-like but passable by some logic? No, actually looking at the deltas, 
    # the player (9) replaces the current cell value.
    # Let's identify the movement logic.
    # In the initial grid, there's a player (9) at r21c36, r22c36, r23c36 (a 3x3 block of color 9).
    # Wait, the player is not a single pixel. It's a 3x3 block of color 9.
    # Looking at the laout:
    # r21c36:9x3, r22c36:9x2, r22c37:4x1, r23c36:9x3.
    # This looks like a "player" entity that is roughly 3x3.
    
    # Find the center of the 3x3 player block (color 9).
    # coords = np.where(grid == 9)
    # If no one is found, it's just returning the same grid.
    #<|channel>thought
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up?
    # Logic: The 3x3 block of color 9 moves in the direction of the action.
    # When it moves, the cells it leaves behind are restored to their 'original' path values.
    # But wait, what was there before? In the deltas, when moving from r21-23 to r27-29, 
    # the old positions (r21-23, c33-35) become 0 and new positions (r27-29, c33-35) become 9.
    # Let's refine:
    # Player is a 3x3 block.
    # Action 2 (Down): Move y + 6? No, r21->r27 is +6.
    # Action 3 (Left): Move x - 6? r33c33 -> r33c27 is -6.
    # Action 4 (Right): Move x + 6? r39c21 -> r39c27 is +6.
    # The player seems to jump by 6 units.
    # The grid structure consists of blocks of 3x3.
    # A "cell" in this maze is actually a 3x3 area.
    # Movement is between these 3x3 cells.
    # ACTION2: Down (y += 6), ACTION3: Left (x -= 6), ACTION4: Right (x += 6).
    # ACTION1: Up (y -= 6)?
    
    # Let's find the current position of the 3x3 player block.
    # We look for color 9.
    # Since it's a 3x3 block, we can take the top-left corner.
    # Note: some pixels might be different colors within the 3x3 block based on initial layout (e.g., r22c37:4x1).
    # But generally, it's centered around color 9.
    
    out = grid.copy()
    
    # Find all indices where value is 9.
    rows, cols = np.where(grid == 9)
    if len(rows) == 0:
        return out
    
    # Approximate top-left of the 3x3 block.
    curr_r = min(rows)
    curr_c = min(cols)
    
    # Movement offsets.
    dr, dc = 0, 0
    if action == 2: dr = 6
    elif action == 3: dc = -6
    elif action == 4: dc = 6
    elif action == 1: dr = -6
    
    new_r = curr_r + dr
    new_c = curr_c + dc
    
    # Boundary checks for a 64x64 grid.
    if new_r < 0 or new_r > 61 or new_c < 0 or new_c > 61:
        return out

    # The player "block" isn't just a solid 3x3 of color 9.
    # It seems to be a pattern that moves.
    # Let's see what happens to the cells it leaves and enters.
    # When moving from (r21, c33) to (r27, c33):
    # Old cells (r21-23, c33-35) become 0.
    # New cells (r27-29, c33-35) become 9.
    # Wait, looking at r29c33: 9x1, 4x1, 9x1. This means the block has internal structure.
    # The most consistent thing is that the 3x3 area changes.
    
    # To implement this simply:
    # 1. Identify the 3x3 region containing the player.
    # 2. Restore the old 3x3 region to some default path value (usually 0).
    # 3. Set the new 3x3 region to the player's pattern.
    # But wait, the deltas show specific values like '4'.
    # Looking at ACTION2 (level 0->0) first transition:
    # r21c33:0x3, r22c33:0x3, r23c33:0x3  (Old pos becomes 0)
    # r27c33:9x3, r28c33:9x3, r29c33:9x1, 4x1, 9x1 (New pos becomes a pattern)
    
    # Let's define the "player pattern" based on the observed transitions:
    # Row 0: [9, 9, 9]
    # Row 1: [9, 9, 9]
    # Row 2: [9, 4, 9]
    
    # Now apply movement.
    # Old position: curr_r, curr_c to curr_r+2, curr_c+2
    for r in range(curr_r, curr_r + 3):
        for c in range(curr_c, curr_c + 3):
            out[r, c] = 0
            
    # New position: new_r, new_c to new_r+2, new_c+2
    pattern = [
        [9, 9, 9],
        [9, 9, 9],
        [9, 4, 9]
    ]
    for i in range(3):
        for j in range(3):
            out[new_r + i, new_c + j] = pattern[i][j]
            
    return out

def is_level_complete(grid):
    """
    Returns True if the level is complete. Usually this means reaching a goal.
    In the observed transitions, there's no win state provided, but typically it's 
    reaching a specific color or coordinate.
    Looking at the grid, maybe reaching the bottom right? Or collecting something?
    Since no WIN STATE was given, we return False unless a clear condition is found.
    Actually, let's check for any cells of color 14 (which might be the goal).
    The initial grid has color 14 at r45-47, c36-38.
    If the player (color 9) overlaps with color 14, it might be complete.
    """
    rows, cols = np.where(grid == 9)
    if len(rows) == 0: return False
    # Check if any part of the player block is on top of the target area (color 14).
    # But wait, the engine replaces values. If the player moves onto 14, 14 is gone.
    # Let's assume completion is when the player reaches the coordinates of the original 14s.
    curr_r = min(rows)
    curr_c = min(cols)
    if curr_r == 45 and curr_c == 33: # Based on last ACTION2 delta r45c33:9x3...
        return True
    return False