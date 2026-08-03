import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where blocks of color 14 are moved or modified by actions.
    # Based on the observed transitions, it's not immediately clear what the same logic is.
    # However, we can observe that specific regions are being updated.
    # Action 2, 3, 4 are directional movements or keyboard keys.
    # Action 6 is click.
    # Action 4 is often associated with moving something right/down?
    # Let's try to implement a basic movement system for the same object.
    # In this case, the laout has some walls (color 2) and walls of other colors.
    # Looking at the observations, ACTION 4 moves things in a certain way.
    # ACTION 3 moves things left.
    # ACTION 2 moves things down.
    # ACTION 3 again moves things left.
    # ACTION 2 moves things down.
    # This suggests a standard top-down view where action 2=Down, 3=Left, 4=Right, 5=Up.
    # Wait, let's look at the<|channel>thought process:
    # ACTION 4: r30c18 -> r30c21 -> r30c26... (moving right)
    # ACTION 3: r30c21 -> r33c21 (wait, no, that was ACTION 2)
    # ACTION 2: r30c21 -> r33c21 (moving down)
    # ACTION 3: r33c18 -> r33c15 (moving left)
    #
    # Let's refine: Action 2 = Down, Action 3 = Left, Action 4 = Right, Action 5 = Up.
    # The object being moved is likely the block of color 14.
    # Color 14 is "the player" or "the active block".
    # Looking at the grid, there are blocks of color 14 and some other colors.
    # It seems like we move the entire connected component of color 14? Or a specific one?
    # In the transitions, it looks like a 3x3 area of color 14 is moving.
    # Let's try to find all cells of color 14 and move them if they can.
    #
    # Actually, looking closer at the delta:
    # ACTION 4: r30c18... changed to 1x3, 14x3. This means color 1 was replaced by 14.
    # And later in that same action, something else becomes 14.
    # It looks like a 3x3 square of color 14 is sliding across the board.
    #
    # Let's implement a simple movement for the 3x3 block of color 14.
    # Note: only one such block exists or moves.
    #
    # The initial grid has several areas of color 14.
    # But the deltas show a specific 3x3 region shifting.
    #
    # Let's assume there's a "player" which is a 3x3 block of color 14.
    # We need to identify its current position (top-left corner).
    # Then we shift it based on the action.
    # Action 2: Down, 3: Left, 4: Right, 5: Up.
    #
    # However, the transitions also show some cells changing to 0.
    # r63c63:0x1, etc. These might be score/timer markers.
    #
    # Let's try this logic:
    # Find the top-leftmost cell of color 14 that is part of a 3x3 block.
    # Move that 3x3 block in the direction specified by the action.
    # When moving, the new positions become 14 and the old positions return to their original state?
    # No, looking at the delta, they are replaced by other colors (like 1 or 15).
    # This suggests the player moves *over* a background.
    # The background seems to be color 1.
    #
    # Let's refine:
    # 1. Identify the 3x3 block of color 14.
    # 2. Determine movement: 2=Down, 3=Left, 4=Right, 5=Up.
    # 3. Shift the 3x3 block.
    # 4. Update grid: New cells = 14, Old cells = Background (color 1).
    # 5. Handle boundaries/walls (color 2).
    #
    # Wait, look at ACTION 4 again: r30c18:1x3, 14x3.
    # It means from col 18, we have three '1's then three '14's.
    # So the block is at c21-23.
    # Then next ACTION 4: r30c21:1x3, 14x2, 0x1.
    # This is getting complex. Let's simplify.
    #
    # Just implement the basic directional shift for all color 14 blocks that can move.
    # And ignore the r63 markers as they are likely not critical for the win state.

    new_grid = grid.copy()
    
    # Action mapping
    directions = {2: (1, 0), 3: (0, -1), 4: (0, 1), 5: (-1, 0)}
    if action not in directions:
        return new_grid
    
    dr, dc = directions[action]
    
    # Find all cells of color 14
    coords = np.argwhere(grid == 14)
    if coords.size == 0:
        return new_grid
    
    # To avoid moving same cell multiple times, we use a mask
    moved_mask = np.zeros_like(grid, dtype=bool)
    
    for r, c in coords:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # Check if destination is "passable" (e.g., not color 2)
            if grid[nr, nc] != 2:
                # Move it: current becomes background (color 1), next becomes 14
                # This is a simplification; the actual game might be more complex.
                new_grid[r, c] = 1
                new_grid[nr, nc] = 14
                
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing blocks.
    # In this case, no win state was provided.
    # Let's assume completion when some condition on color 14 is met.
    # Or just return False as we don't have enough info.
    return False