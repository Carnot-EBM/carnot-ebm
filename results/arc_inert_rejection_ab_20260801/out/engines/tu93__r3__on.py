import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle where an object (color 9 or similar)
    # moves through a corridor of color 0 and 2/5.
    # The actions are likely directional movement for the same entity.
    # Action 2: Down, Action 3: Left, Action 4: Right, Action 1: Up.
    
    new_grid = grid.copy()
    H, W = new_grid.shape
    
    # Find the "player" / moving piece. In the initial state, it's at r21-23, c33-35.
    # We look for blocks of color 9.
    # player_pos = None
    # For simplicity, we find all cells of color 9.
    p_cells = np.argwhere(grid == 9)
    if p_cells.size == 0:
        return new_grid
    
    # Assume the player is a 3x3 block.
    # Use the first cell to identify the top-left corner.
    r_min, c_min = p_cells[0]
    r_max, c_max = p_cells[-1]
    
    # Define directions based on observed transitions:
    # ACTION 2: Down (r+3)
    # ACTION 3: Left (c-6)
    # ACTION 4: Right (c+6)
    # ACTION 1: Up (r-3)
    
    dr, dc = 0, 0
    if action == 2:
        dr, dc = 3, 0
    elif action == 3:
        # In the observations, Action 3 moves it from c33 to c27 then to c21.
        # Each step is 6 columns.
        dr, dc = 0, -6
    elif action == 4:
        # In the observations, Action 4 moves it from c21 to c27 then to c33.
        dr, dc = 0, 6
    elif action == 1:
        dr, dc = 0, -3 # This is a bit ambiguous but usually 1=Up
        # Wait, looking at the data: r21->r27 (Action 2), r39->r45 (Action 2).
        # Let's correct directions.
        # dr = -3 for Action 1, dr = 3 for Action 2? No, observed: r21 -> r27 is +6 rows.
        # Actually, let's look closer: r21c33 -> r27c33 is ACTION 2. That's +6 rows.
        # r33c33 -> r33c27 is ACTION 3. That's -6 cols.
        # r39c21 -> r39c27 is ACTION 4. That's +6 cols.
        # So: 1=Up (-6), 2=Down (+6), 3=Left (-6), 4=Right (+6).
    
    # Corrected Directions
    if action == 2:
        dr, dc = 6, 0
    elif action == 3:
        dr, dc = 0, -6
    elif action == 4:
        dr, dc = 0, 6
    elif action == 1:
        dr, dc = -6, 0

    # Move the player block (all cells of color 9)
    # Check if move is possible (destination must be color 0 or same as current)
    # We identify the target area.
    # The "player" is a 3x3 block.
    #<|channel>thought
    # Let's just apply the movement and update the grid.
    
    # Find all cells that are currently color 9.
    mask_9 = (grid == 9)
    # Clear old position.
    new_grid[mask_9] = 0 # This might be wrong; it should return to original background.
    # But in observations, r21c33 becomes 0x3, etc.
    # So we set them to 0.
    
    # Shift mask.
    shifted_mask = np.zeros_like(mask_9, dtype=bool)
    for r, c in zip(*np.where(mask_9)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < H and 0 <= nc < W:
            shifted_mask[nr, nc] = True
            
    # Set new position to 9.
    new_grid[shifted_mask] = 9
    
    # Special case for the "counter" at r63.
    # In every transition, r63c55 -> r63c54 -> r63c52...
    # It seems a cell of color 0 is being placed or removed from the right side of row 63.
    # Let's track the counter cells (color 0).
    # The observed deltas show r63c55:0x1, then r63c54:0x1, etc.
    # This means as the player moves, the grid at r63 fills with 0s from right to left.
    
    # Find current count of 0s at the end of row 63.
    # Start from column 63 and go backwards.
    count_0 = 0
    for c in range(W-1, -1, -1):
        if grid[63, c] == 0:
            count_0 += 1
        else:
            break
    
    # Increment count of 0s by 1 or 2 depending on move?
    # Action 2: r63c55:0x1 (+1)
    # Action 2: r63c54:0x1 (+1)
    # Action 3: r63c52:0x2 (+2)
    # Action 3: r63c51:0x1 (+1)
    # Action 2: r63c50:0x1 (+1)
    # Action 4: r63c49:0x1 (+1)
    # Action 4: r63c47:0x2 (+2)
    # Action 2: r63c46:0x1 (+1)
    # It seems the counter just increments based on some logic. Let's just mirror the observed delta if possible.
    # Actually, looking at the deltas, it's simply filling row 63 with 0s from right to left.
    # We can just find the first non-zero cell from the right and set it to 0.
    
    # Find the same "counter" behavior.
    for c in range(W-1, -1, -1):
        if new_grid[63, c] != 0:
            new_grid[63, c] = 0
            break

    return new_grid

def is_level_complete(grid):
    # Level complete when player reaches a certain target or counter fills.
    # No win state provided, but usually it's reaching a goal (color 14).
    # In initial grid, color 14 is at r45-47, c39-41.
    # If any part of the player (color 9) overlaps with color 14, it might be complete.
    p_cells = np.argwhere(grid == 9)
    g_cells = np.argwhere(grid == 14)
    if p_cells.size == 0 or g_cells.size == 0:
        return False
    
    # Check for overlap.
    for pr, pc in p_cells:
        for gr, gc in g_cells:
            if pr == gr and pc == gc:
                return True
    return False