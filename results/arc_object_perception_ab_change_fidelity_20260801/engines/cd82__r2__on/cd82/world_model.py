import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a movement-based puzzle where an agent/object moves through a maze or environment.
    # Based on the observed transitions, ACTION4 and ACTION2 move something.
    # ACTION4 (down) and ACTION2 (left) likely correspond to directional movements.
    # ACTION2: Left, ACTION4: Down, ACTION6: Click.
    # In the same world model, we<|channel>thought that// No analysis allowed in comments.
    # Let's implement a simple movement logic based on the observed delta changes.
    # # Action mapping: 1: Up, 2: Left, 3: Right, 4: Down.
    
    # We need to identify the "player" object. Looking at the initial grid, there's a lot of color 5 (background).
    # Color 2 is often associated with the moving object.
    # In the deltas, cells change from 5 to 2, and others from 2 to 5.
    # This suggests a block of color 2 is moving.
    
    # Find all coordinates of the player object (color 2).
    # Since there are multiple objects, let's assume the one that moves is the primary actor.
    # The most common moving part is color 2.
    # 15 is also appearing in the deltas.
    # 15 is often used as a 'collected' or 'transformed' state.
    # 15 might be an item to collect.
    
    # Based on the transitions:
    # ACTION4: Moves something down/right? No, looking at r21c39... it seems like a complex movement.
    # Let's look at the delta again.
    # r21c39:2x1 -> cell (21, 39) becomes 2.
    # r63c63:5x1 -> cell (63, 63) becomes 5.
    # It looks like a "snake" or a "block" is shifting.
    
    # Actually, the observed transitions show a specific pattern of cells changing values.
    # Let's try to implement a simple translation for the block of color 2.
    
    # Find all pixels of color 2.
    # If no color 2 exists, we can't move it.
    # 15 is also moving.
    # The object consists of colors 2 and 15.
    # This suggests a composite object.
    
    # Identify the "active" object: connected components of {2, 15}.
    # We will treat any pixel of color 2 or 15 as part of the player.
    
    player_pixels = np.argwhere(np.isin(grid, [2, 15]))
    if len(player_pixels) == 0:
        return grid.copy()

    # Action mapping based on typical ARC-AGI directional actions:
    # 1: Up, 2: Left, 3: Right, 4: Down.
    # In the provided deltas:
    # ACTION4 (Down): r21c39... r63c63 becomes 5. It seems like something shifted down/right.
    # ACTION2 (Left): r21c39:5x1... r45c38:2x9. Something shifted left/down.
    
    # Let's refine the action map:
    # 1: Up (-1, 0), 2: Left (0, -1), 3: Right (0, 1), 4: Down (1, 0).
    
    dy, dx = 0, 0
    if action == 1: dy, dx = -1, 0
    elif action == 2: dy, dx = 0, -1
    elif action == 3: dy, dx = 0, 1
    elif action == 4: dy, dx = 1, 0
    else: return grid.copy()

    new_grid = grid.copy()
    
    # To avoid overwriting pixels we are moving, we store current player positions.
    # We move only those that are part of the "player" object.
    # The observed deltas show a complex shift. Let's try simple translation first.
    
    for y, x in player_pixels:
        # Clear old position if it's not being filled by another pixel
        # This is tricky for blocks. Simple way: clear all then draw all.
        pass
    
    # Correct approach for block movement:
    # 1. Identify player mask.
    # 2. Create new mask shifted by (dy, dx).
    # 3. Fill background where old mask was and player colors where new mask is.
    
    mask = np.isin(grid, [2, 15])
    shifted_mask = np.zeros_like(mask)
    
    # Shift mask
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if mask[y, x]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    shifted_mask[ny, nx] = True
    
    # Update grid
    # First, set all current player pixels to background color 5.
    # Then, set the shifted positions to their original values.
    # Note: this doesn't handle "collision" or "collection".
    # But looking at deltas, it seems like a simple shift of the whole object.
    
    # To preserve colors within the moving object:
    temp_colors = grid[mask]
    new_grid[mask] = 5 # Background
    
    # Find coordinates of shifted mask
    new_coords = np.argwhere(shifted_mask)
    # We need to map old coords to new coords to keep colors.
    # This is only possible if the shift is 1-to-1.
    
    # Let's try a simpler approach for the specific observed transitions.
    # The deltas show that when the object moves, some parts change from 2 to 15 or vice versa.
    # This looks like a "snake" where the head changes color or something similar.
    # However, without more data, a simple translation is the best general rule.
    
    # Re-mapping colors:
    old_coords = np.argwhere(mask)
    for (oy, ox), color in zip(old_coords, temp_colors):
        ny, nx = oy + dy, ox + dx
        if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
            new_grid[ny, nx] = color

    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or collecting all items.
    # In this game, it might be moving the block of color 2/15 to a certain area.
    # Looking at the INITIAL GRID, there are areas of color 0 (black).
    # Maybe the goal is to cover those? Or move the player into them?
    # Without a WIN STATE example, we'll assume completion when no color 0 remains.
    return not np.any(grid == 0)