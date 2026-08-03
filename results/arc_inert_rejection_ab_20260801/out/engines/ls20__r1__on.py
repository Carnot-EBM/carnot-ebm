import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Inducing rules from observed transitions:
    # Action 1: Move Up
    # Action 2: Move Down
    # Action 3: Move Left
    # Action 4: Move Right
    # Action 5: Move Diagonal Up-Left
    # Action 6: Click/Interact
    # Action 7: Other
    
    # The game seems to be a puzzle where an object (represented by color 3 or similar)
    # and some target areas are defined.
    # The same cells that change in the la//s20/world_model.py
    # Let's look at the same coordinates of changes.
    # Actions 1, 3, 4 move something.
    # # We identify the "player" position based on the changing cells.
    # Based on the transition deltas, it looks like the player is represented by 
    # the value 3 at r61c14..r62c19.
    # # Let's find the current position of the 'cursor' or 'player'.
    # cursor_pos = np.where(grid == 3)
    # But wait, the grid contains many 3s.
    # Looking at the<|channel>thought process, we can actually see that theC
    # # Let's find the 3s that are moving.
    # In the INITIAL GRID, r61c14:3x1 and r62c14:3x1 are 3s.
    # After ACTION3 (Left), they move to c15? No, Action 3 is usually Left.
    # Let's re-examine:
    # Initial: r61c14:3x1, r62c14:3x1
    # ACTION3: r61c14:3x1, r62c14:3x1 -> r61c15:3x1, r62c15:3x1 (Wait, this is Right?)
    # ACTION3: r61c15:3x1, r62c15:3x1 -> r61c16:3x1, r62c16:3x1 (No, ACTION1 is Up)
    # ACTION1 (Up): r61c16:3x1, r62c16:3x1 -> r61c17:3x1, r62c17:3x1...
    # This mapping is weird. Let's look at the cursor again.
    # The cells changing in r61 and r62 are always 3s.
    # ACTION3 moves them from c14 to c15? No, that's right.
    # ACTION1 moves them from c16 to c17? That's also right.
    # Wait, let's check ACTION4 (Right).
    # ACTION4 moves them from c19 to c20.
    # It seems Action 1, 3, 4 all move the cursor RIGHT in these examples?
    # No, let's re-read carefully.
    # INITIAL: r61c14:3x1, r62c14:3x1
    # ACTION3: changed cells = r61c14:3x1, r62c14:3x1 ... wait "changed cells" means NEW values.
    # So r61c14 becomes 3? But it was already 3.
    # Actually, if a cell is listed as "r61c14:3x1", it means grid[61, 14] = 3.
    # If it was 3 and stays 3, it wouldn't be in the delta unless the whole span changed.
    # Let's look at the cursor movement again.
    # Init: r61c14=3, r62c14=3
    # Action 3: r61c15=3, r62c15=3 (and r61c14, r62c14 must have become something else)
    # Action 3: r61c16=3, r62c16=3... no, that was Action 1.
    # This is very confusing. Let's simplify.
    # The player moves a block of colors in the main area (rows 8-49).
    # The blocks are usually color 3, 12, 9.
    # When ACTION1 happens, the block moves UP.
    # When ACTION3 happens, the block moves LEFT? No, looking at c24 -> c19, it moved LEFT.
    # When ACTION4 happens, the block moves RIGHT.
    # So: Action 1 = Up, Action 3 = Left, Action 4 = Right.
    # And there's a cursor at the bottom (r61, r62) that tracks this movement.
    # Cursor X increases when block moves right, decreases when left.
    # Cursor Y doesn't seem to change much.
    # Block position:
    # Init: rows 45-49, cols 19-23 (approx)
    # After Action 3 (Left): block moves from c24 to c19.
    # After Action 1 (Up): block moves from r40-44 to r35-39.
    # This is a standard "move object" game.

    # Let's implement the same logic as seen in deltas.
    # We need to find the 'object' (the cluster of non-background colors).
    # Background is color 4 and 5.
    # The object consists of colors like 3, 9, 12.
    # Since we only have a few transitions, let's just move the object based on action.
    
    # Find current object bounds
    mask = np.isin(grid, [3, 9, 12])
    coords = np.argwhere(mask)
    if coords.size == 0:
        return grid
    
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    # Object shape
    obj_slice = grid[min_r:max_r+1, min_c:max_c+1]
    
    # Clear old position
    new_grid = grid.copy()
    new_grid[min_r:max_r+1, min_c:max_c+1] = 4 # Assume background is 4
    
    # Determine new position
    dr, dc = 0, 0
    if action == 1: dr = -5  # Move Up (based on r40->r35)
    elif action == 2: dr = 5   # Move Down
    elif action == 3: dc = -5  # Move Left (based on c24->c19)
    elif action == 4: dc = 5   # Move Right (based on c19->c24)
    
    nr, nc = min_r + dr, min_c + dc
    
    # Boundary checks (simple)
    nr = max(0, min(nr, 63 - obj_slice.shape[0]))
    nc = max(0, min(nc, 63 - obj_slice.shape[1]))
    
    # Place object at new position
    new_grid[nr:nr+obj_slice.shape[0], nc:nc+obj_slice.shape[1]] = obj_slice
    
    # Update cursor at the bottom (rows 61, 62)
    # The cursor is a pair of 3s.
    cursor_mask = np.where((grid == 3) & (np.arange(64)[:, None] >= 61))
    if cursor_mask[0].size > 0:
        old_cx = cursor_mask[1][0]
        new_cx = old_cx + (1 if dc > 0 else -1 if dc < 0 else 0)
        new_cx = max(0, min(63, new_cx))
        # Clear old cursor
        for r in [61, 62]:
            if grid[r, old_cx] == 3:
                new_grid[r, old_cx] = 11 # Background for cursor area
        # Set new cursor
        for r in [61, 62]:
            new_grid[r, new_cx] = 3

    return new_grid

def is_level_complete(grid):
    # No win state provided, but typically it's when the object reaches a target.
    # For now, return False as we don't have the win condition.
    return False