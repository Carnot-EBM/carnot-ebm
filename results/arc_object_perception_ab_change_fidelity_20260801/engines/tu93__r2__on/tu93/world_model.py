import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up (inferred)
    
    # The game seems to be a puzzle where a specific "player" or "active" object moves.
    # Based on the same-shape tracking and delta analysis:
    # ACTION2 -> r21c33:0x3... r27c33:9x3... (Object at (21,33) moved to (27,33))
    # ACTION3 -> r33c27:9x3... r33c33:0x3... (Object at (33,27) moved to (33,33)? No, wait.)
    # Let's re-examine the deltas.
    # ACTION2: Object at (21, 33) becomes color 0, object at (27, 33) becomes color 9.
    # ACTION3: Object at (33, 27) becomes color 9, object at (33, 33) becomes color 0.
    # ACTION4: Object at (39, 21) becomes color 0, object at (39, 27) becomes color 9.
    # It looks like a token (color 9) is moving between fixed positions (grid cells of size 3x3).
    # The grid is divided into 3x3 blocks.
    # Token position is current block index (row_block, col_block).
    # Action moves the token to an adjacent block if possible.
    # There are "walls" or "obstacles" represented by colors other than 5 (background).
    # However, the observed transitions show the token (color 9) moving and changing existing blocks.
    # Actually, looking closer at the same-shape tracking:
    # r63c55:0x1... etc. This suggests some progress bar or counter at the bottom.
    # Let's find the token (color 9).
    # INITIAL GRID has no color 9 except in r21-r23, c33-c35? No, wait.
    # Initial Grid: r21: ... 9x3 ... (at col 33), r22: ... 9x2,4x1 ... (at col 33), r23: ... 9x3 ... (at col 33).
    # So the token starts at row 21-23, col 33-35.
    
    # Find the token (color 9)
    token_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                token_pos = (r, c)
                break
        if token_pos: break

    if token_pos is None:
        return grid.copy()

    # The token is a 3x3 block of color 9.
    # Token center is roughly (row // 3 * 3 + 1, col // 3 * 3 + 1)
    tr, tc = token_pos[0], token_pos[1]
    br, bc = tr // 3 * 3, tc // 3 * 3
    
    # Target position based on action
    dr, dc = 0, 0
    if action == 1: dr, dc = -3, 0 # Up
    elif action == 2: dr, dc = 0, 6 # Wait, ACTION2 moved it from row 21 to 27? That's Down.
    elif action == 3: dr, dc = 0, -6 # Left
    elif action == 4: dr, dc = 0, 6 # Right
    
    # Let's re-map actions based on the observed transitions:
    # Transition 1: ACTION2 moves r21c33 -> r27c33. (Down)
    # Transition 2: ACTION2 moves r27c33 -> r33c33. (Down)
    # Transition 3: ACTION3 moves r33c33 -> r33c27. (Left)
    # Transition 4: ACTION3 moves r33c27 -> r33c21. (Left)
    # Transition 5: ACTION2 moves r33c21 -> r39c21. (Down)
    # Transition 6: ACTION4 moves r39c21 -> r39c27. (Right)
    # Transition 7: ACTION4 moves r39c27 -> r39c33. (Right)
    # Transition 8: ACTION2 moves r39c33 -> r45c33. (Down)
    
    # Correct Action Mapping:
    # ACTION1: Up
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    
    if action == 1: dr, dc = -6, 0
    elif action == 2: dr, dc = 0, 6 # Wait, the deltas say ACTION2 is DOWN. Let's use that.
    elif action == 3: dr, dc = 0, -6
    elif action == 4: dr, dc = 0, 6
    
    # Re-evaluating ACTION2 again: r21c33 to r27c33 is row +6.
    if action == 1: dr, dc = -6, 0
    elif action == 2: dr, dc = 6, 0
    elif action == 3: dr, dc = 0, -6
    elif action == 4: dr, dc = 0, 6

    new_br, new_bc = br + dr, bc + dc
    
    # Check if target block exists and is within bounds
    if not (0 <= new_br < grid.shape[0] and 0 <= new_bc < grid.shape[1] and 
            0 <= new_br + 2 < grid.shape[0] and 0 <= new_bc + 2 < grid.shape[1]):
        return grid.copy()

    # The token moves by replacing the existing colors in the 3x3 area.
    # The old position becomes color 5 (background) or whatever was there?
    # No, looking at deltas: "r21c33:0x3" -> it became color 0.
    # Let's see what was there before the move.
    # In INITIAL GRID, r21-23 c33-35 were color 9.
    # After ACTION2, they become color 0.
    # And r27-29 c33-35 become color 9.
    # It seems the token (color 9) swaps its color with the destination block.
    
    new_grid = grid.copy()
    
    # Old position values to restore later if needed, but the observed transitions show a specific pattern.
    # The target block is already some color.
    # The same-shape tracking says objects are moving.
    # Actually, let's just implement the swap of color 9 and the other color.
    
    # Find all cells of color 9
    token_cells = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                token_cells.append((r, c))
    
    # Target cells for the new 3x3 block
    target_cells = []
    for r in range(br + dr, br + dr + 3):
        for c in range(bc + dc, bc + dc + 3):
            target_cells.append((r, c))
            
    # What does the old position become?
    # In Transition 1: r21c33 (color 9) -> color 0.
    # In Transition 2: r27c33 (color 9) -> color 0.
    # In Transition 3: r33c33 (color 9) -> color 0.
    # It seems the token always leaves behind color 0.
    
    for r, c in token_cells:
        new_grid[r, c] = 0 # Or maybe it restores background? Let's use 0 as observed.
        # Wait, looking at INITIAL GRID, some blocks are already color 0 or 2.
        # The token moves and changes colors. This is a "painting" game.
    
    for r, c in target_cells:
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            new_grid[r, c] = 9
            
    # Also need to handle the progress bar at the bottom (r63).
    # ACTION2: r63c55:0x1... etc.
    # Each move reduces the length of the color 6 line at the bottom.
    # The color 6 line starts at r63, c0-55.
    # As we move, the end of the line (at col 55) shrinks.
    # Transition 1: r63c55 becomes 0.
    # Transition 2: r63c54 becomes 0.
    # Transition 3: r63c52 becomes 0 (two cells!).
    # Transition 4: r63c51 becomes 0.
    # Transition 5: r63c50 becomes 0.
    # Transition 6: r63c49 becomes 0.
    # Transition 7: r63c47 becomes 0 (two cells!).
    # Transition 8: r63c46 becomes 0.
    # It seems every move consumes some amount of "fuel" or "energy" from the bar.
    
    # Find the current end of the color 6 bar.
    bar_end = -1
    for c in range(grid.shape[1]-1, -1, -1):
        if grid[63, c] == 6:
            bar_end = c
            break
    
    if bar_end != -1:
        # How much to consume? The deltas show varying amounts (1 or 2).
        # Let's assume it consumes 1 per move for now.
        new_grid[63, bar_end] = 0
        
    return new_grid

def is_level_complete(grid):
    # Level complete when the token reaches a target or the fuel bar is empty.
    # Or maybe when all blocks are painted?
    # In this game, we don't have a win state provided.
    # Usually, it's reaching a specific coordinate or painting everything.
    # Given the observed transitions, let's check if the token reached r45-47, c33-35.
    # That was the last transition.
    token_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 9:
                token_pos = (r, c)
                break
        if token_pos: break
    
    if token_pos is None: return False
    
    # Check if it reached the goal area (color 14 block at r45-47, c33-35).
    # The INITIAL GRID has color 14 at r45-47, c33-35.
    # If the token (color 9) moves onto it, that might be the win condition.
    return token_pos[0] >= 45 and token_pos[1] >= 33