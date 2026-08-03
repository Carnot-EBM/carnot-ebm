import numpy as np

def engine(grid, action, data):
    """
    Induces the transition rules for game 'ka59'.
    Based on the provided logs:
    - ACTION2, ACTION3, ACTION4 are directional movements (likely Up, Left, Right).
    - There is an entity (a 3x3 block of color 1) moving across the grid.
    - When it moves into cells of color 14, those cells change to color 1.
    - When it leaves cells, they might revert or stay modified depending on the rule.
    - Looking at the deltas:
      ACTION4 (Right): Shifts a 3x3 block of 1s rightward, replacing 14s with 1s.
      ACTION3 (Left): Shifts a 3x3 block of 1s leftward, replacing 14s with 1s.
      ACTION2 (Up): Shifts a 3x3 block of 1s upward, replacing 14s with 1s.
    - The "entity" seems to be the contiguous block of color 1.
    """
    new_grid = grid.copy()
    
    # Find the current position of the player/block (color 1)
    # We assume the 'player' is the primary connected component of color 1.
    coords = np.argwhere(grid == 1)
    if coords.size == 0:
        return new_grid

    # Determine center of the 3x3 block
    center_y = int(np.mean(coords[:, 0]))
    center_x = int(np.mean(coords[:, 1]))
    
    # Define movement vectors for actions
    # Based on typical ARC mappings and observed deltas:
    # ACTION2: Up, ACTION3: Left, ACTION4: Right, ACTION5: Down (inferred)
    dy, dx = 0, 0
    if action == 2: dy = -1
    elif action == 3: dx = -1
    elif action == 4: dx = 1
    elif action == 5: dy = 1
    else: return new_grid # Other actions not observed to move the block

    # Calculate new boundaries for a 3x3 block
    new_cy, new_cx = center_y + dy, center_x + dx
    
    # The logic from deltas suggests that as the block moves, it 'paints' color 1
    # over existing colors (like 14). It doesn't necessarily erase its previous trail.
    # However, looking at "r30c21:1x3 r31c21:1x3 r32c21:1x3" in ACTION2, 
    # it seems the old position is cleared back to something else or just shifted.
    
    # To simulate the movement of a 3x3 block:
    # 1. Clear current block if it's a strict move (though some ARC games leave trails)
    # In this specific game, the deltas show the old cells becoming 14 again or changing.
    # Let's implement a shift and paint mechanism.
    
    for y in range(center_y - 1, center_y + 2):
        for x in range(center_x - 1, center_x + 2):
            if 0 <= y < 64 and 0 <= x < 64:
                # If we are moving away from here, restore to background/wall?
                # The logs suggest the 'player' moves and leaves color 1 behind 
                # but also clears parts. This is complex without more data.
                # Simple approach: Move the 3x3 block.
                pass

    # Based on observed delta "r30c21:1x3 r31c21:1x3 r32c21:1x3" for ACTION2,
    # it seems the column at c=21 was set back to 1.
    # Actually, looking closer: Action 4 shifted something right.
    # Let's apply the movement of the 3x3 block of 1s.
    
    # Clear old position (approximate)
    for y in range(center_y - 1, center_y + 2):
        for x in range(center_x - 1, center_x + 2):
            if 0 <= y < 64 and 0 <= x < 64:
                # Restore to a default value if it was part of the player
                # In ka59, the walls are 2, paths are 14/15.
                new_grid[y, x] = grid[y, x] # Placeholder
    
    # Paint new position
    for y in range(new_cy - 1, new_cy + 2):
        for x in range(new_cx - 1, new_cx + 2):
            if 0 <= y < 64 and 0 <= x < 64:
                # Only paint if not hitting a wall (color 2)
                if new_grid[y, x] != 2:
                    new_grid[y, x] = 1

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the block reaches a target or clears all color 14s.
    Given no win state provided, we assume completion based on common ARC patterns.
    """
    # Check if any specific goal condition is met (e.g., no more 14s left)
    return np.sum(grid == 14) == 0