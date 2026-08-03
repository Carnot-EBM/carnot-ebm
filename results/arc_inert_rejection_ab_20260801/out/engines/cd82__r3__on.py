import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where objects move based on actions.
    # Based on the observed transitions, ACTION 2 and 4 are movement/interaction.
    # Action 2 is likely 'Up' or 'Down', Action 4 is 'Right'.
    # Action 6 is click.
    
    # Since we only have very limited transition data and no clear pattern of
    # simple physics, we will implement a basic simulation of the same changes.
    # But since this is ans ARC-AGI task, we uma<|channel>thought
    # This specific level cd82 looks like it involves moving a "player" object (color 2)
    # through a maze or environment.
    
    # Identify player position
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 2:
                player_pos = (r, c)
                # We assume there is only one player object (or a cluster of color 2)
                # break
    
    # If multiple cells are color 2, we find the center of mass or first occurrence.
    # In the initial grid, color 2 appears at r24c25 to r32c37 approx.
    # The transitions show that the block of color 2 moves.
    # Let's refine the search for the 'active' block of color 2.
    
    coords_2 = np.argwhere(grid == 2)
    if len(coords_2) == 0:
        return grid.copy()

    # Action mapping based on common ARC patterns:
    # ACTION 2: Up/Down? (Transitions show changes in row indices)
    # ACTION 4: Right? (Transitions show changes in col indices)
    # ACTION 6: Click.
    
    # Based on observed deltas:
    # ACTION 4: r21c39... r37c43. This looks like it shifted the block of 2s right and down.
    # ACTION 2: r21c39... r45c38. This looks like it shifted the block of 2s down.
    
    # We will implement a simple movement rule:
    # ACTION 2 -> Down
    # ACTION 4 -> Right
    # ACTION 1 -> Left
    # ACTION 3 -> Up
    
    dr, dc = 0, 0
    if action == 2: # Down
        dr = 1
        dc = 0
    elif action == 4: # Right
        dr = 0
        dc = 1
    elif action == 1: # Left
        dr = 0
        dc = -1
    elif action == 3: # Up
        dr = -1
        dc = 0

    new_grid = grid.copy()
    
    # Find all cells that are color 2 (the player/block)
    player_cells = np.argwhere(grid == 2)
    
    # To avoid "smearing", we first clear old positions if they aren't being filled
    # For this specific game, the transitions show complex shape changes.
    # However, without more data, the most general rule is shifting.
    
    # Let's try to shift the block of 2s and handle collisions with walls (color 4).
    # Color 5 is background.
    
    for r, c in player_cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # If target is not a wall (color 4), move it.
            # But observed deltas show color 2 moving into areas that were color 15 or 5.
            # We will simply move them.
            new_grid[nr, nc] = 2
            # Only clear original if it's not occupied by another moved cell
            # This is tricky for blocks.
    
    # Clear cells that are no longer part of the shifted block
    # In a real engine, you'd calculate the new set of coords first.
    new_coords = []
    for r, c in player_cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            new_coords.append((nr, nc))
    
    # To properly shift a block:
    # 1. Identify all current 'player' cells.
    # 2. Calculate their new positions.
    # 3. Set old positions to background (color 5) unless they are now new positions.
    # 4. Set new positions to player color (2).
    
    final_grid = grid.copy()
    # Use color 5 as default background for clearing
    for r, c in player_cells:
        final_grid[r, c] = 5
        
    for nr, nc in new_coords:
        final_grid[nr, nc] = 2

    return final_grid

def is_level_complete(grid):
    # Win state usually involves reaching a goal or collecting items.
    # In the observed data, we don't have a win state grid.
    # We will return False by default.
    return False