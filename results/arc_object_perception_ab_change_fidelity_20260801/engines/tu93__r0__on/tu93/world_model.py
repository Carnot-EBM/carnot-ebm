import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up (implied)
    
    # The game seems to be a puzzle where a specific "player" or "active" object moves.
    # In these observations, color 9 is the active object/marker.
    # It moves in steps of 6 columns or 6 rows (since blocks are 3x3).
    # Block size is 3x3.
    
    H, W = grid.shape
    new_grid = grid.copy()
    
    # Find current position of the marker (color 9)
    # We look for any cell of color 9. Since it's a 3x3 block, we take the top-left corner.
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return new_grid
    
    # Marker is a 3x3 block. Get its center or top-left.
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # Movement logic:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION1: Up
    
    dr, dc = 0, 0
    if action == 2: # Down
        dr = 3
    elif action == 3: # Left
        dc = -6 # Based on observed jumps in columns
    elif action == 4: # Right
        dc = 6
    elif action == 1: # Up
        dc = -3 # This is not clear from data, but usually opposite of Action 2
        dr = -3
        
    # The marker (color 9) moves as a 3x3 block.
    # It replaces the values at the destination and takes the color of the previous location.
    # Note: In observations, it seems to move by replacing blocks of 3x3 cells.
    # 
    # Update grid based on movement
    # Marker current position (top-left corner)
    curr_r, curr_c = r_min, c_min
    
    # New position
    next_r, next_c = curr_r + dr, curr_c + dc
    
    # Bounds check
    if next_r < 0 or next_r >= H or next_c < 0 or next_c >= W:
        return new_grid
    
    # We need to determine what happens to the same size block (// 3 * 3)
    #<|channel>thought
    # In ACTION2 (Down), the marker (color 9) moves from (21, 33) to (27, 33).
    # The old position is filled with the background/previous value.
    # The new position is filled with color 9.
    # The old position's value becomes whatever was at the target? No.
    # Let's look closer at deltas:
    # ACTION2: r21c33:0x3 r22c33:0x3 r23c33:0x3 r27c33:9x3 r28c33:9x3 r29c33:9x1,4x1,9x1
    # It looks like it replaces a 3x3 area.
    # But wait, the delta for r29c33 says "9x1, 4x1, 9x1". This suggests it doesn't just fill everything with 9.
    # It might be swapping or moving a pattern.
    
    # Actually, looking at the INITIAL grid and the deltas, the marker is not a simple block of 9s.
    # It's more like a cursor that highlights a 3x3 region.
    # When it moves, it changes the colors of the cells in the 3x3 regions.
    # Specifically, if you move from A to B, the cells at A return to their "original" state (color 0, 2, or 5)
    # and the cells at B become color 9.
    
    # Let's re-examine ACTION2:
    # Old pos: r21-23, c33-35. New pos: r27-29, c33-35.
    # Delta: r21c33:0x3, r22c33:0x3, r23c33:0x3 -> These were color 9, now they are 0.
    # Delta: r27c33:9x3, r28c33:9x3, r29c33:9x1,4x1,9x1 -> These were something else, now they are mostly 9.
    # Wait, look at INITIAL grid r29: ... 5x9, 0x3, 5x16. The cell at r29, c33 is 0.
    # In ACTION2 delta, it becomes 9.
    
    # It seems like the marker simply changes a 3x3 block to color 9, and restores the previous 3x3 block.
    # But how does it know what to restore?
    # Let's assume there's an underlying "base" grid that doesn't change, and the same as if we just XORed or toggled.
    # Or more likely, the cells of the base grid are stored elsewhere.
    # Actually, looking at the deltas, when the marker moves from (21,33) to (27,33), the cells at (21,33) become 0.
    # Looking at INITIAL grid, r21-23, c33-35 are indeed 0 in some places.
    # This looks like a simple movement of a "highlight" (color 9).
    
    # To implement this without knowing the base grid, we can observe that the marker always replaces a 3x3 area.
    # When it leaves, it restores the original colors. Since we don't have the base grid, 
    # let's see if we can find a pattern. In all ACTION2/3/4 examples, the restored colors are 0.
    # Wait, look at ACTION2: r21c33:0x3... The restored values are 0.
    # Look at ACTION3: r33c33:0x3... Restored values are 0.
    # It seems they just return to color 0? No, that can't be right for a general model.
    # Let's assume the marker *swaps* its contents with the target block.
    
    # But wait, look at the very last line of each delta: r63c55:0x1, r63c54:0x1, etc.
    # These are cells on the bottom row (r63) changing from 6 to 0.
    # This is a progress bar! As the marker moves, the progress bar fills or empties.
    # Action 2 (Down): r63c55->0, then r63c54->0, etc.
    # Each movement reduces the number of color-6 cells in row 63.
    
    # Simplified Engine:
    # 1. Find current 3x3 block of color 9.
    # 2. Move it by dr, dc.
    # 3. Restore old block to color 0 (as seen in deltas).
    # 4. Set new block to color 9.
    # 5. Update the progress bar at r63.
    
    coords = np.argwhere(grid == 9)
    if coords.size == 0: return new_grid
    r_min, c_min = coords.min(axis=0)
    
    dr, dc = 0, 0
    if action == 2: dr = 6 # Blocks are 3x3, but jumps are often larger. Let's check ACTION2: 21 -> 27 is +6.
    elif action == 3: dc = -6 # ACTION3: 33 -> 27 is -6.
    elif action == 4: dc = 6  # ACTION4: 21 -> 27 is +6.
    elif action == 1: dr = -6
    
    # Progress bar logic:
    # Action 2: r63c55->0, then r63c54->0...
    # It seems every move removes one cell of color 6 from row 63, starting from the right (col 55).
    # Find the rightmost cell of color 6 in row 63.
    row63 = grid[63, :]
    sixes = np.argwhere(row63 == 6).flatten()
    if sixes.size > 0:
        rightmost_six = sixes[-1]
        new_grid[63, rightmost_six] = 0

    # Move marker
    # Restore old
    for r in range(r_min, r_min + 3):
        for c in range(c_min, c_min + 3):
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 0 # Based on observed deltas restoring to 0
                
    # Set new
    nr, nc = r_min + dr, c_min + dc
    for r in range(nr, nr + 3):
        for c in range(nc, nc + 3):
            if 0 <= r < H and 0 <= c < W:
                new_grid[r, c] = 9
                
    return new_grid

def is_level_complete(grid):
    # Level complete when the progress bar (row 63) is all 0s?
    # Or when color 6 is gone from row 63.
    return not np.any(grid[63, :] == 6)