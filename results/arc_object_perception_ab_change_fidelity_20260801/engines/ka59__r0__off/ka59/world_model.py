import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The observed transitions show a player object moving and interacting with other elements.
    # Based on the same pattern of changes in thes deltas, it's a movement/interaction game.
    # Action 2: Down, Action 3: Left, Action 4: Right, Action 1: Up
    # Let's identify the player position based on the current state.
    # In this initial grid, therethoughtout patterns suggest a player might be represented by color 1 or 14.
    # The observed actions are 2, 3, 4 correspond to Down, Left, Right.
    # 
    # Looking at the delta for ACTION 4 (Right):
    # r30c18:1x3,14x3 -> cells at (30, 18-20) become 1, (30, 21-23) become 14.
    # This suggests a player is moving from one cell to the_next.
    # Find the player (color 14).
    # 
    # Given the observations, the laout looks like a maze own with walls (color 2).
    #<|channel>thought// No analysis allowed in comments.
    
    # Simplified logic based on observation:
    # Player is likely color 14.
    # Movement: Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right.
    # Walls are color 2.
    
    player_pos = np.argwhere(grid == 14)
    if len(player_pos) == 0:
        return grid
    
    # We assume there's only one main "player" entity or we move all of them.
    # The deltas show blocks of 14s moving together.
    # Let's find the bounding box of the 14s.
    rows, cols = np.where(grid == 14)
    if len(rows) == 0: return grid
    
    min_r, max_r = np.min(rows), np.max(rows)
    min_c, max_c = np.min(cols), np.max(cols)
    
    dr, dc = 0, 0
    if action == 1: dr = -1
    elif action == 2: dr = 1
    elif action == 3: dc = -1
    elif action == 4: dc = 1
    
    new_grid = grid.copy()
    
    # Move the block of 14s
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r, c] == 14:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    # Check for wall (color 2)
                    if grid[nr, nc] != 2:
                        new_grid[nr, nc] = 14
                        # If we move from a cell, it might become color 1 or something else.
                        # The deltas show cells becoming 1x3 then 14x3.
                        # This suggests trailing colors.
                        pass

    # Clean up old positions
    # This is tricky without knowing exactly what replaces the player.
    # In ACTION 4: r30c18:1x3,14x3 means current pos becomes 1, next becomes 14.
    # Let's try to implement this "trail" logic.
    
    # To avoid complex state tracking, let's just apply the movement of the block.
    # # Re-evaluating based on ACTION 4 delta: r30c18:1x3,14x3
    # Original was likely 14 at c18-20, now it's 1 at c18-20 and 14 at c21-23.
    
    # Find all blocks of 14s
    mask_14 = (grid == 14)
    new_mask_14 = np.zeros_like(mask_14)
    
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if mask_14[r, c]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and grid[nr, nc] != 2:
                    new_mask_14[nr, nc] = True
                else:
                    # If blocked, stay put
                    new_mask_14[r, c] = True

    # Update grid: old 14s that didn't move become 1s? Or just everything moves.
    # Let's use a simpler approach: if the block can move, shift it and leave color 1 behind.
    
    can_move = True
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r, c] == 14:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] == 2:
                    can_move = False
                    break
        if not can_move: break
    
    if can_move:
        # Move all 14s to new positions and set old positions to 1
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                if grid[r, c] == 14:
                    new_grid[r + dr, c + dc] = 14
                    new_grid[r, c] = 1
    else:
        return grid

    return new_grid

def is_level_complete(grid):
    # No win state provided, but typically it's reaching a goal or clearing something.
    # Based on the deltas, maybe color 0 appearing?
    return np.any(grid == 0)