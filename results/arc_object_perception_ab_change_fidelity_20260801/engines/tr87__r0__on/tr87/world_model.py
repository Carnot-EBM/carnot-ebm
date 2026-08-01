import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game involves moving a "cursor" or "player" represented by color 4 at (63, 62) initially.
    # Action 4 seems to be move cursor horizontally.
    # Action 2 and 1 seem to interact with blocks in the bottom area.
    #
    # Based on observed transitions:
    # ACTION 4 moves the player (color 4) leftward.
    # ACTION 2/1 might shift colors in the region r52-r56.
    # Let's refine based on the delta observations:
    # ACTION 4: r63c62 -> r63c61 -> r63c60 -> r63c59...
    # la also changes cells in r48, r49, r59, r60.
    # These are rows 48, 49, 59, 60 which contain color 0 (black) gaps.
    # It looks like the "gaps" of color 0 are shifting horizontally as the player moves.
    #
    # ACTION 1: Shifts some pixels in r52-r56.
    # ACTION 2: Shifts some pixels in r52-r56.
    #
    # The goal is likely to align something or clear a path.
    #
    # Since we only have a few frames and no own logic for the same time step,
    # let's implement the movement of the cursor and the gap shifts.

    new_grid = grid.copy()
    
    # Find the player (color 4)
    player_pos = np.where(grid == 4)
    if len(player_pos[0]) > 0:
        py, px = player_pos[0][0], player_pos[1][0]
        
        if action == 4: # Move Left
            # Move player
            new_grid[py, px] = grid[py, px - 1] if px > 0 else grid[py, px]
            new_grid[py, px - 1] = 4
            
            # Shift gaps in rows 48, 49, 59, 60
            gap_rows = [48, 49, 59, 60]
            for row in gap_rows:
                # This is a specific pattern shift observed in deltas
                # For r48 and r60, it's a span of 5 cells.
                # For r49 and r59, it's a single cell.
                # In ACTION 4, we see shifts like r48c15:3x5 -> r48c22:3x5...
                # The same idea is that the "active" part of these rows is moving with the cursor.
                pass

    # Action 1 & 2 are likely vertical or other interactions.
    # la’t implement them as a<|channel>thought process but since I must return code:
    # Let's just handle the player movement for now based on the provided transitions.
    
    # Find current player position to determine direction
    player_pos = np.where(grid == 4)
    if len(player_pos[0]) > 0:
        py, px = player_pos[0][0], player_pos[1][0]
        if action == 4: # Move Left
            new_grid[py, px] = grid[py, px - 1] if px > 0 else grid[py, px]
            new_grid[py, px - 1] = 4
            
            # Shift gaps in r48, r49, r59, r60
            # In ACTION 4, we see color 3 (blue) replacing color 0 (black) and vice versa.
            # We find all cells of color 0 in those rows and shift them left by some amount?
            # No, it looks like they move in blocks.
            for row in [48, 49, 59, 60]:
                # Simple approximation: shift everything in that row left by 7 pixels?
                # The delta shows c15 -> c22 is a jump of 7.
                # Wait, the cursor moves from c62 to c61 (delta 1).
                # But the gap shifts from c15 to c22 (delta +7).
                # This means as player moves LEFT, gaps move RIGHT.
                pass

    # Let's try to implement the observed deltas more literally for Action 4.
    if action == 4:
        player_pos = np.where(grid == 4)
        if len(player_pos[0]) > 0:
            py, px = player_pos[0][0], player_pos[1][0]
            new_grid[py, px] = grid[py, px - 1] if px > 0 else grid[py, px]
            new_grid[py, px - 1] = 4
            
            # Shift color 0 and 3 blocks in rows 48, 49, 59, 60
            for r in [48, 49, 59, 60]:
                row_data = grid[r].copy()
                # Find indices of color 0
                zeros = np.where(row_data == 0)[0]
                # Move them right by 7?
                for z in zeros:
                    if z + 7 < 64:
                        new_grid[r, z+7] = 0
                        new_grid[r, z] = row_data[z-7] if z >= 7 else 3
    
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when the player reaches a goal or clears objects.
    # Let's assume it's not complete yet.
    return False