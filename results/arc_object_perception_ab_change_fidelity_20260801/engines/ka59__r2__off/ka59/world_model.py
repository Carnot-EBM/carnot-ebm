import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a movement/interaction puzzle where certain colors act as walls or interactable objects.
    # Based on the observed transitions, it's not clear what the exact rules are.
    # However, we can see that patterns of color 14 and 1 move around or shift.
    #<|channel>thought
    # Let's analyze the movements.
    # Action 2: Up? Action 3: Right? Action 4: Down?
    # Action 1: Left?
    # Looking at the deltas:
    # ACTION4 (Down) shifts something from row 30-32 to 33-35.
    # ACTION3 (Right) shifts things horizontally.
    # ACTION2 (Up) might actually be moving an object "up" in some context, but here it shifted r30-32 to r33-35 which is DOWN.
    # Wait, let's re-examine.
    # Initial Grid has structures at rows 21-41.
    # Transition 1: ACTION4 -> changes cells at r30c18...
    # Transition 2: ACTION4 -> changes cells at r30c21...
    # Transition 3: ACTION4 -> changes cells at r30c26...
    # Transition 4: ACTION3 -> changes cells at r30c21...
    # Transition 5: ACTION2 -> changes cells at r30c21 and r33-35.
    # This looks like a cursor or a block of colors moving around.
    # Let's assume action mapping: 2=Up, 3=Left, 4=Down, 1=Right? No, usually 2=Up, 3=Down, 4=Left, 1=Right or similar.
    # In many ARC games: 2=Up, 3=Down, 4=Left, 1=Right. Or 2=Up, 3=Right, 4=Down, 1=Left.
    # Let's try the standard WASD/Arrow keys map for these integers if possible.
    # Action 2 (r30->r33) is Down.
    # Action 3 (c21->c18 then c18->c15) is Left.
    # Action 4 (changes in place or shifts slightly) might be Up or Right.
    # Actually, looking at "ACTION4" deltas, they are modifying existing blocks.
    # The most consistent pattern is that an object (a 3x3 block of color 14/1) moves.
    # Object start position: r30-32, c18-20 approx.
    # ACTION4: modifies it.
    # ACTION3: moves it left (c21->c18, c18->c15).
    # ACTION2: moves it down (r30-32 -> r33-35).
    # This suggests: 2=Down, 3=Left, 4=Up? No, let's look at the coordinates again.
    # Transition 5: ACTION2 moves something from r30-32 to r33-35. That is DOWN.
    # Transition 6 & 7: ACTION3 moves something from c21 to c18 and then c18 to c15. That is LEFT.
    # So 2=Down, 3=Left. Then likely 1=Right, 4=Up.
    # But wait, ACTION4 was used first and didn't move the block rows, just changed colors inside.
    # Let's re-read carefully.
    # Initial grid has a lot of color 2 (background), color 15 (walls?), color 1 (player/object?).
    # The "changed cells" are the key.
    # In transition 5: ACTION2 changes r30c21...r32c21 to 1x3 (color 1) AND r33c21...r35c21 to 14x3 (color 14).
    # This means the object at r30-32 moved to r33-35.
    # In transition 6: ACTION3 changes r33c18...r35c18 to 14x3. It was at c21. Now it's at c18. Moved LEFT.
    # In transition 7: ACTION3 changes r33c15...r35c15 to 14x3. Moved LEFT again.
    # So: Action 2 = Down, Action 3 = Left.
    # Then likely: Action 1 = Right, Action 4 = Up.
    # Let's check if Action 4 moves things up. Transition 1, 2, 3 use Action 4 but they don't shift rows. They change colors.
    # Maybe Action 4 is "Interact" or "Up"? If it's "Up", and there's a wall, it might just toggle color?
    # Actually, looking at the very first few transitions:
    # ACTION4 (level 0->0): changed cells r30c18:1x3,14x3 ...
    # This looks like it's changing the internal state of the block.
    # Given the limited data, the most robust model is that an object moves in the grid.
    # The object is a 3x3 block.
    # We need to find the current position of this block (color 14) and move it.

    # Find the 3x3 block of color 14
    coords = np.argwhere(grid == 14)
    if coords.size == 0:
        return grid
    
    # Assume the block is the same size and shape throughout
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Movement vectors
    moves = {
        2: (1, 0),  # Down
        3: (0, -1), # Left
        1: (0, 1),  # Right
        4: (-1, 0), # Up
    }
    
    dy, dx = moves.get(action, (0, 0))
    
    # Create new grid
    new_grid = grid.copy()
    
    # Erase old block (set to background color 2 or whatever was there)
    # This is tricky because we don't know the exact "background" inside the puzzle area.
    # But based on deltas, they replace with color 1.
    for r in range(y_min, y_max + 1):
        for c in range(x_min, x_max + 1):
            new_grid[r, c] = 1 if grid[r, c] == 14 else 2

    # Move and place new block
    ny_min, nx_min = y_min + dy, x_min + dx
    ny_max, nx_max = y_max + dy, x_max + dx
    
    # Boundary checks
    if ny_min < 0 or ny_max >= grid.shape[0] or nx_min < 0 or nx_max >= grid.shape[1]:
        return grid # Block same position if hit boundary
        
    # Check for walls (color 15)
    for r in range(ny_min, ny_max + 1):
        for c in range(nx_min, nx_max + 1):
            if grid[r, c] == 15:
                return grid # Hit wall
                
    # Place the block
    for r in range(ny_min, ny_max + 1):
        for c in range(nx_min, nx_max + 1):
            new_grid[r, c] = 14
            
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when an object reaches a target.
    # Since we don't have one, return False.
    return False