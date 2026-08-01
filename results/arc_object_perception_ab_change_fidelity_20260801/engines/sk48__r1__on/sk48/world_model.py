import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where an agent moves a 'player' object (color 6) and collects items.
    # Action 1: Move Up
    # Action 2: Move Down
    # Action 3: Move Left/Right? No, based on observed transitions, ACTION 3 changes color 8, 14, 9 blocks.
    # Action 4: Interact/Toggle?
    # Let's refine movement based on the observed deltas.
    
    # Find player (color 6)
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 6:
                player_pos = (r, c)
                # We only need one representative point of the player object
                break
        if player_pos: break

    # If no player found, return original grid
    if not player_pos:
        return grid.copy()

    # Simple movement logic induced from observations
    # Based on the same-shape objects, we
    # find all cells belonging to the player object (the shape id 3ffcc6441eb802ba).
    # The bounding box of the player is usually a 6x6 block or similar.
    #<|channel>thought
    # In the INITIAL GRID, obj7 is color 6 bbox=(24, 11, 29, 16) px=18.
    # Wait, looking at the run-length encoding for ACTION 1 (level 0->0):
    # r18c11:6x6 ... r23c11:6x6. This looks like the player moved UP.
    # Action 1: Move Up
    # Action 2: Move Down
    # Action 3: Interact with items?
    # Action 4: Toggle/Interact?
    # Action 5: Left?
    # Action 6: Right?
    # Let's try to map actions to directions based on deltas.
    # ACTION 1: Player moves from y=24..29 to y=18..23. Delta = -6 rows.
    # ACTION 2: Player moves from y=18..23 back to y=24..29, then maybe further down.
    # ACTION 2 again: Player moves from y=24..29 to y=30..35. Delta = +6 rows.
    # So Action 1 is UP, Action 2 is DOWN.
    
    # The grid contains 'walls' or 'obstacles' (color 4).
    # Movement is blocked by color 4.
    
    # Find all cells of the player object
    player_cells = np.argwhere(grid == 6)
    if len(player_cells) == 0:
        return grid.copy()
    
    # Bounding box of the player
    y0, x0 = np.min(player_cells[:, 0]), np.min(player_cells[:, 1])
    y1, x1 = np.max(player_cells[:, 0]), np.max(player_cells[:, 1])
    
    # Define movement delta
    dy, dx = 0, 0
    if action == 1: dy = -6
    elif action == 2: dy = 6
    elif action == 3: # Based on observations, this changes items in a specific area
        # This looks like it might be "collect" or "activate"
        # Let's simulate ACTION 3 as changing colors of blocks at y=19..22, x=36..45 to something else.
        # In observed transitions, Action 3 changed r19c36:8x4 etc. (color 8, 4).
        # It seems to replace color 8 with 4 and vice versa? Or just set them to 4/8.
        # Looking at INITIAL GRID: obj4 is color 8 bbox=(19, 42, 22, 45), obj9 is color 9...
        # Actually, let's look at the deltas for ACTION 3 again.
        # r19c36:8x4, r19c42:4x4 ... these are replacing existing things.
        # For now, we will implement movement and basic interaction if possible.
        pass
    elif action == 4: # Interact / Toggle
        # Observed ACTION 4 changes cells at r20c23, r21c23, etc.
        # These are small patterns of colors 2, 1.
        pass

    # Execute Movement
    if dy != 0 or dx != 0:
        new_grid = grid.copy()
        # Clear old player position
        for r, c in player_cells:
            new_grid[r, c] = 5 # Background color
        
        # Calculate new bounding box
        ny0, nx0 = y0 + dy, x0 + dx
        ny1, nx1 = y1 + dy, x1 + dx
        
        # Check boundaries and collisions with color 4 (walls)
        if ny0 < 0 or ny1 >= grid.shape[0] or nx0 < 0 or nx1 >= grid.shape[1]:
            return grid.copy()
        
        # Simple collision check: any cell in the same shape as player moving into a wall?
        # For simplicity, we'll just move it if the target area is not mostly walls.
        # In this game, movement seems to be in steps of 6.
        
        # Re-draw player object at new position
        # The player object isn't just a block; it has internal structure.
        # We need to preserve that structure.
        for r, c in player_cells:
            nr, nc = r + dy, c + dx
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                new_grid[nr, nc] = grid[r, c]
        
        return new_grid

    return grid.copy()

def is_level_complete(grid):
    # Level complete usually means all items collected or player reached goal.
    # Based on observed data, there are no "win" grids provided, but typically
    # color 0 (black) might disappear or certain colors change.
    # Let's assume completion when specific objects are gone.
    # Since we don't have a win state, return False by default unless something obvious happens.
    return False