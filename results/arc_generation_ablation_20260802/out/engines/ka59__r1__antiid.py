import numpy as np

def engine(grid, action, data):
    # Induce rules based on the same set of transitions.
    # The provided observations show ACTION4 (Right/Down?), ACTION3 (Left), ACTION2 (Up).
    # It seems like theres a moving object (a 3x3 block of color 1)
    # Let's identify the current position of the 3x3 block of color 1.
    
    # Find the center of the 3x3 block of color 1
    rows, cols = np.where(grid == 1)
    if len(rows) == 0:
        return grid
    
    # We assume the<|channel>thought
    # Looking at the delta changes, the block moves and replaces colors.
    # The block is 3x3.
    # Center coordinates
    center_r, center_c = int(np.median(rows)), int(np.median(cols))
    
    # Define movement vectors for actions
    # Action 2: Up, Action 3: Left, Action 4: Right/Down? (Actually looking at deltas, it looks like ACTION 4 might be 'right')
    # Based on observed deltas:
    # ACTION 4 (level 0->0): r30c18:1x3,14x3 ... -> Block moved to c=18-20
    # ACTION 4 again: r30c21:1x3... -> Block moved to c=21-23
    # ACTION 4 again: r30c26:14x1,1x3 ... -> Block moved to c=26-28
    # ACTION 3 (level 0->0): r30c21:14x3,1x3 ... -> Block moved back to c=21-23 but replaced by color 14? No, the delta says "1x3" is new.
    # Let's re-examine: "r30c18:1x3,14x3" means cells from col 18 to 20 become 1 and 21 to 23 become 14.
    # This suggests a block of size 3 moves and leaves/picks up colors.
    
    # Movement mapping
    move_map = {
        2: (-1, 0), # Up
        3: (-1, 0), # Wait, Action 3 in deltas shows movement along columns (Left)
        4: (0, 1),  # Right
    }
    
    # Correcting based on observed transitions:
    # ACTION 4: Moves right (cols increase)
    # ACTION 3: Moves left (cols decrease)
    # ACTION 2: Moves up (rows decrease) - wait, ACTION 2 delta: r33, r34, r35... rows increased! So ACTION 2 is Down.
    
    # Revised move map
    move_map = {
        2: (1, 0),  # Down
        3: (0, -1), # Left
        4: (0, 1),  # Right
    }

    if action not in move_map:
        return grid

    dr, dc = move_map[action]
    
    # The object is the 3x3 block of color 1.
    # Find all cells of color 1.
    mask_1 = (grid == 1)
    coords_1 = np.argwhere(mask_1)
    if len(coords_1) == 0:
        return grid
    
    # Get bounding box of color 1
    min_r, min_c = coords_1.min(axis=0)
    max_r, max_c = coords_1.max(axis=0)
    
    # Create a new grid
    new_grid = grid.copy()
    
    # Fill old position with color 14 (as seen in deltas "14x3")
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            new_grid[r, c] = 14
            
    # Move and fill new position with color 1
    new_min_r, new_min_c = min_r + dr, min_c + dc
    new_max_r, new_max_c = max_r + dr, max_c + dc
    
    # Boundary check
    if new_min_r < 0 or new_max_r >= grid.shape[0] or new_min_c < 0 or new_max_c >= grid.shape[1]:
        return grid

    for r in range(new_min_r, new_max_r + 1):
        for c in range(new_min_c, new_max_c + 1):
            new_grid[r, c] = 1
            
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a goal is reached.
    # For now, return False as we don't have the target.
    return False