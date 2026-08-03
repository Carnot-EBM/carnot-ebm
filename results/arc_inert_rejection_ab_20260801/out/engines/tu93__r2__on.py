import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle where an object (color 9 or 4)
    # moves across a grid of obstacles/paths.
    # The "object" seems to be color 9 (or sometimes 4 for a single cell).
    # Action 2: Down
    # Action 3: Left
    # Action 4: Right
    # Action 1: Up
    # Action 5: Not seen in deltas but likely Up
    # Let's refine based on ACTION2=Down, ACTION3=Left, ACTION4=Right.
    # Action 1 would then be Up.
    
    # Find current position of the 'player' (the object that changes)
    # We assume the player is represented by color 9.
    # Since there are multiple cells of color 9, we's treat it as a 3x3 block.
    # 3x3 blocks of color 9 are moving.
    # Looking at the same delta: r21c33:0x3 r22c33:0x3 r23c33:0x3 r27c33:9x3 r28c33:9x1,4x1,9x1 r29c33:9x3...
    # The "object" is a 3x3 block of color 9.
    # 3x3 blocks of color 9/4.
    
    # Identify all 3x3 blocks of color 9 or 4.
    # For simplicity, let's find the center of the 3x3 block.
    # Let's look for the first occurrence of color 9.
    #<|channel>thought
    # In ACTION2 (Down), the block moves from (21, 33) to (27, 33).
    # In ACTION3 (Left), the block moves from (33, 33) to (33, 27).
    # In ACTION4 (Right), the block moves from (33, 21) to (33, 27).
    # Then (33, 27) to (33, 33).
    # Action 2: Down (y+6)
    # Action 3: Left (x-6)
    # Action 4: Right (x+6)
    # Action 1: Up (y-6)
    
    # Find current position of the player object (the 3x3 block of color 9).
    # We search for a 3x3 area that is not background (color 5).
    # The "player" is the only thing moving.
    # The cells it leaves behind are restored to their original state or some other value.
    # Looking at the deltas, the cells it leaves become 0s or something else.
    # Let's find where the 9s are.
    
    # Find all coordinates of color 9.
    coords = np.argwhere(grid == 9)
    if coords.size == 0:
        return grid
    
    # Assume the player is the top-left corner of the 3x3 block.
    # Use the minimum row and column.
    r_min, c_min = coords[0] if coords.size > 0 else (0, 0)
    # But wait, there might be multiple blocks of 9.
    # In the INITIAL GRID, r21c33 is 9x3... no, r21 has 9x3 starting at col 33?
    # No, let's re-read: r21:5x21,0x3,5x3,0x3,2x3,9x3,2x3,0x3,2x3,0x3,5x16
    # Col indices for r21: 0..20(5), 21..23(0), 24..27(5), 28..30(0), 31..33(2), 34..36(9).
    # So color 9 starts at col 34 in r21.
    
    # Let's find the center of the mass of color 9.
    y, x = np.mean(coords, axis=0).astype(int)
    
    # Movement vectors
    move_map = {
        1: ( -6, 0 ), # Up
        2: ( 6, 0 ),  # Down
        3: ( 0, -6 ), # Left
        4: ( 0, 6 ),  # Right
    }
    
    if action not in move_map:
        return grid
    
    dy, dx = move_map[action]
    new_y, new_x = y + dy, x + dx
    
    # We need to know what was there before and restore it.
    # This is tricky without a map. But looking at deltas:
    # ACTION2: r21c33:0x3 ... r27c33:9x3.
    # It seems the player "swaps" or moves into a slot.
    # The slots are pre-defined by the layout.
    # Color 5 is background. Colors 0, 2, 14 are obstacles/walls.
    # Let's assume the movement only happens if the target 3x3 block is 'empty' (color 0).
    
    # To implement this correctly, we should find all cells of color 9,
    # shift them, and then fill the old position with whatever was there?
    # No, look at the delta: r21c33 becomes 0x3.
    # In INITIAL GRID, r21c33 was 9x3. So 9 -> 0.
    # Then r27c33 becomes 9x3. In INITIAL GRID, r27c33 was 0x3. So 0 -> 9.
    # This means it's a simple swap between color 9 and color 0 in 3x3 blocks.
    
    # Find the current 3x3 block of 9s.
    # Since there might be multiple, let's just move ALL 9s that form a 3x3 block.
    # Or more simply, find any cell of color 9 and its surrounding 3x3.
    
    new_grid = grid.copy()
    
    # Identify the player block (the one moving)
    # We can find the top-left corner of the 3x3 block of 9s.
    # Let's search for the first 9 and assume it's part of the player.
    coords = np.argwhere(grid == 9)
    if coords.size == 0: return grid
    
    # To handle potential other 9s, we look for the most "compact" group or just the first.
    # Given the game, there's likely only one player.
    r_start, c_start = coords[0]
    # The actual top-left might be slightly different if coords[0] is not the top-left.
    # But since we use min, it should work.
    r_min, c_min = np.min(coords, axis=0)
    
    # Move the 3x3 block
    for dr in range(3):
        for dc in range(3):
            old_r, old_c = r_min + dr, c_min + dc
            new_r, new_c = old_r + dy, old_c + dx
            
            if 0 <= new_r < grid.shape[0] and 0 <= new_c < grid.shape[1]:
                # Swap values
                val_at_new = grid[new_r, new_c]
                new_grid[old_r, old_c] = val_at_new
                new_grid[new_r, new_c] = 9
                
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing something.
    # In this game, look at the bottom right: r63c56...
    # The deltas show cells at r63 changing from 0 to something else?
    # No, "r63c55:0x1" means cell (63, 55) becomes 0.
    # This looks like a progress bar or a set of keys being collected.
    # Let's assume completion when all cells in a certain area are color 0.
    # Or simply check if any specific condition is met.
    # Given no win state provided, we can't be sure.
    # But often it's about moving the player to a goal.
    # Let's guess that completing the level means the player reaches the same position as some target.
    # For now, return False unless a clear pattern emerges.
    return False