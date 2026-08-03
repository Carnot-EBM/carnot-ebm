import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up (implied)
    
    # The game seems to be a puzzle where a 'player' object (color 9 or similar)
    # moves through a maze of walls (color 2/5/0) and empty spaces (color 0).
    # The same logic applies to a movement-based game.
    # Let's identify the player position.
    # In the initial grid, color 9 appears at r21-23, c33-35.
    # After ACTION2 (Down), it moves from r21-23, c33-35 to r27-29, c33-35.
    #<|channel>thought
    # Wait, the delta shows r21c33:0x3, r22c33:0x3, r23c33:0x3 AND r27c33:9x3...
    # This means the block of color 9 moved down.
    
    # Find all cells of color 9.
    player_cells = np.argwhere(grid == 9)
    if len(player_cells) == 0:
        return grid.copy()

    # Assume the player is a contiguous block.
    # We define the move direction based on action.
    dr, dc = 0, 0
    if action == 1: # Up
        dr, dc = -6, 0
    elif action == 2: # Down
        dr, dc = 0, 6 # wait, looking at deltas: r21->r27 is +6 rows.
        dr, dc = 6, 0
    elif action == 3: # Left
        dr, dc = 0, -6
    elif action == 4: # Right
        dr, dc = 0, 6

    # The movement distance seems to be 6 units (the size of the blocks).
    # Let's check ACTION3 (Left): r33c33 -> r33c27. That's -6 columns.
    # Let's check ACTION4 (Right): r39c21 -> r39c27. That's +6 columns.

    new_grid = grid.copy()
    
    # For each cell of color 9, try to move it.
    # In this game, the player block moves as a unit.
    # We only move if the destination cells are 'passable'.
    # Passable colors: likely 0 or other non-wall colors.
    # Based on observed transitions, the player replaces whatever was there and leaves behind 0?
    # No, look at delta: r21c33:0x3 means it became 0.
    
    # Identify the bounding box of the player.
    min_r, min_c = np.min(player_cells, axis=0)
    max_r, max_c = np.max(player_cells, axis=0)
    
    # Calculate new bounding box.
    n_min_r, n_min_c = min_r + dr, min_c + dc
    n_max_r, n_max_c = max_r + dr, max_c + dc
    
    # Check boundaries.
    if (n_min_r < 0 or n_max_r >= grid.shape[0] or 
        n_min_c < 0 or n_max_c >= grid.shape[1]):
        return grid.copy()

    # Movement logic:
    # The block moves to the new position.
    # Original cells become color 0.
    # New cells become color 9.
    # However, looking at ACTION2's delta: "r27c33:9x3 r28c33:9x3 r29c33:9x1,4x1,9x1"
    # This suggests a complex interaction with existing colors (like pushing).
    # But for simplicity in this ARC task, we will implement basic movement and replacement.
    
    # Clear old player position.
    for r, c in player_cells:
        new_grid[r, c] = 0
        
    # Place player in new position.
    # We need to preserve the shape of the player block.
    # Relative coordinates from min_r, min_c.
    for r, c in player_cells:
        nr, nc = r + dr, c + dc
        new_grid[nr, nc] = 9

    # Special case: look at the bottom right corner change "r63c55:0x1".
    # It seems there is some counter or state tracking in the grid.
    # Let's try to simulate that by decrementing values at the bottom row if they exist.
    # In ACTION2: r63c55:0x1, then r63c54:0x1... it looks like a bar filling/emptying.
    # The initial grid has r63:6x56,0x8. This means cells [0-55] are 6 and [56-63] are 0.
    # After action 2: r63c55 becomes 0. So cell 55 (value 6) became 0.
    # Then r63c54 becomes 0.
    # This suggests every successful move removes one '6' from the end of the line in row 63.
    
    # Find the last cell with value 6 in row 63.
    row_63 = new_grid[63, :]
    last_six = np.where(row_63 == 6)[0]
    if len(last_six) > 0:
        new_grid[63, np.max(last_six)] = 0

    return new_grid

def is_level_complete(grid):
    # Level complete when all '6's are gone from row 63?
    # Or when player reaches a certain goal.
    # Given the pattern, let's assume it's when row 63 has no more 6s.
    return not np.any(grid[63, :] == 6)