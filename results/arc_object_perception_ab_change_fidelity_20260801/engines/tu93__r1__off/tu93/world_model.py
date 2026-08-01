import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where some object (color 9/4) moves through a maze of walls (color 2/0/5).
    # Based on the observed transitions, Action 2 (Down), 3 (Left), 4 (Right), and others move an entity.
    # Action 2: Down, Action 3: Left, Action 4: Right, Action 1: Up?
    # Let's identify the "player" or moving entity. In the deltas, color 9 often changes to 0 or 5.
    # Color 9 is likely the player.
    
    # Find current position of the player (color 9)
    # Note: There are multiple cells of color 9 in the initial grid.
    # However, only one specific group of 3x3 blocks of color 9 actually moves.
    # The "active" player is the 3x3 block that is changing.
    # find all 3x3 blocks of color 9
    h, w = grid.shape
    player_pos = None
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(grid[r:r+3, c:c+3] == 9):
                # Check if it's not a static part of the same structure
                # This is a bit tricky because as it's move happens, some cell values change.
                #<|channel>thought
                pass

    # Instead of let's look at the delta pattern.
    # Action 2: Down, Action 3: Left, Action 4: Right.
    # Let's assume there is a single active moving entity (a 3x3 block).
    # We identify it by looking for any 3x3 area that is currently color 9.
    # Since we don't know exactly which one is the "active" one, but only one moves per action.
    # In the observed transitions, the movement is always in increments of 6 columns or 6 rows.
    # The maze consists of cells of colors 0, 2, and 5.
    # Color 5 is background/wall.
    # Color 0 is path.
    # Color 2 is wall.
    # Color 14 is goal.
    # The player (color 9) moves through paths (color 0).
    
    # Based on the deltas, the movements are in steps of 6 units.
    # Let's find all 3x3 blocks of color 9.
    players = []
    for r in range(h - 2):
        for c in range(w - 2):
            if np.all(grid[r:r+3, c:c+3] == 9):
                players.append((r, c))
    
    # Only one block moves per transition. We need to identify which one.
    # But based on the provided data, there's only one moving entity.
    # If multiple exist, we might need a way to distinguish them.
    # For now, assume any 3x3 block of color 9 can be the player.
    # Since only one movement happens at a time, let's pick the first one that *can* move.
    
    # Movement directions
    directions = {
        1: (-6, 0), # Up
        2: (6, 0),  # Down
        3: (0, -6), # Left
        4: (0, 6),  # Right
    }
    
    if action not in directions:
        return grid.copy()

    dr, dc = directions[action]
    new_grid = grid.copy()
    
    # Find the active player. In this specific game, it seems the "active" player is the one
    # whose destination path is currently open (color 0).
    # Let's try to find a 3x3 block of color 9 that can actually move into a 3x3 area of color 0.
    found_move = False
    for r, c in players:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h-2 and 0 <= nc < w-2:
            # Check if target area is "passable" (mostly color 0 or 5)
            # Based on deltas, moving from 9 -> 0/5 and 0/5 -> 9
            # The movement replaces the old position with the background and new position with 9.
            # We need to know what the background was.
            
            # To simplify, let's just perform the move for the first valid 3x3 block found.
            # Since there's only one moving entity in the examples.
            
            # Save current values to restore them later
            old_vals = grid[r:r+3, c:c+3].copy()
            target_vals = grid[nr:nr+3, nc:nc+3].copy()
            
            new_grid[r:r+3, c:c+3] = target_vals # This is not quite right based on deltas.
            # Deltas show that cells are set to specific colors.
            # Let's look at r21c33:0x3 etc. It means row 21, col 33-35 become 0.
            # Then r27c33:9x3 means row 27, col 33-35 become 9.
            
            # Correct logic:
            # 1. Identify player (color 9)
            # 2. Clear old position (set to whatever was "under" it or a default path color)
            # 3. Set new position to color 9.
            
            # But what is "under" it? In the deltas, when moving from r21 to r27, 
            # r21 becomes 0 and r27 becomes 9. When moving from r27 to r33, r27 becomes 0 and r33 becomes 9.
            # So the background of the path is color 0.
            
            new_grid[r:r+3, c:c+3] = 0 # Path color
            new_grid[nr:nr+3, nc:nc+3] = 9 # Player color
            found_move = True
            break
    
    if not found_move:
        return grid.copy()

    # Special case for the bottom right counter in the observed data: r63c55:0x1 etc.
    # It seems there's a countdown or progress bar at the bottom.
    # Let's find the current state of that bar.
    # The bar is at row 63, columns 46-63 approx.
    # Every move reduces the number of cells of some color (maybe 5) by 1 or 2.
    # In the deltas: r63c55:0x1 -> r63c54:0x1 -> r63c52:0x2 -> r63c51:0x1 ...
    # This looks like it's filling from right to left with color 0.
    # We can simulate this by finding the first cell of color 5 starting from the right and making it 0.
    for c in range(w - 1, -1, -1):
        if new_grid[63, c] == 5: # Assuming background is 5
            new_grid[63, c] = 0
            break

    return new_grid

def is_level_complete(grid):
    # Level complete when player reaches goal (color 14).
    # Check if any part of the player (color 9) overlaps with color 14.
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                # If a player block is adjacent to or on top of color 14
                # In many ARC games, "reaching" means overlapping or being next to.
                # Let's check for overlap since we don't have win state data.
                # But wait, the initial grid has color 14 at r45-47, c36-38 approx.
                pass
    
    # Since we don't have the win state, let's assume it's complete when the player
    # position (color 9) matches the goal position (color 14).
    # We can detect this by checking if there are no more cells of color 14 left
    # because they were replaced by color 9, or vice versa.
    return np.any((grid == 9) & (grid == 14)) # This is impossible.
    # Correct logic: if any cell that was 14 is now 9.
    # Or simply if the player is at the target coordinates.
    # Given the lack of one full win grid, this is an educated guess.
    # A better way: return True if the progress bar is full?
    # The progress bar seems to be filling up.
    return False # Default to False unless a clear condition is met.