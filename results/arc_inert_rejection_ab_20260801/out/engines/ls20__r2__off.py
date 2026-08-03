import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right
    # The game seems to be a puzzle where some blocks move in groups or are shifted.
    # Based on the deltas, it's a bit complex but the same object moves.
    # ACTION1: shifts something up
    # ACTION3: shifts something left
    # ACTION4: shifts something right
    
    # Identify the "player" or "active object" - looking at the colors 12 and 9
    # In the initial grid, color 12 is at r45-46, c24-28. Color 9 is at r47-49, c24-28.
    # This is a<|channel>thought
    # Let's find the bounding box of the active object (colors 12 and 9)
    # coords = np.argwhere((grid == 12) | (grid == 9))
    # if len(coords) == 0: return grid
    # 
    # Bounding box
    # y_min, x_min = coords.min(axis=0)
    # y_max, x_max = coords.max(axis=0)
    # 
    # The movement seems to be additive.
    # For ACTION1 (Up), we shift the block of cells with values 12/9 upwards.
    #  uma own copy of the same shape as the grid
    # new_grid = grid.copy()
    # 
    # Looking at the deltas:
    # ACTION3 (Left): shifted from c24 to c19. Delta -5 columns.
    # ACTION4 (Right): shifted from c19 back to c24. Delta +5 columns.
    # ACTION1 (Up): shifts blocks up by 5 rows.
    # 
    # It also affects some other pixels in r61-62.
    # Let's implement a simple translation for the active object and the marker.
    
    new_grid = grid.copy()
    
    # Find all coordinates of colors 12 and 9
    active_coords = np.argwhere((grid == 12) | (grid == 9))
    if len(active_coords) == 0: return new_grid
    
    y_min, x_min = active_coords.min(axis=0)
    y_max, x_max = active_coords.max(axis=0)
    
    # Movement offsets
    dy, dx = 0, 0
    if action == 1: dy = -5
    elif action == 2: dy = 5
    elif action == 3: dx = -5
    elif action == 4: dx = 5
    
    # The "marker" is color 3 at r61c14..20.
    # Let's find it.
    marker_coords = np.argwhere(grid == 3)
    # Filter markers to only those in rows 61-62
    marker_mask = (marker_coords[:, 0] >= 61) & (marker_coords[:, 0] <= 62)
    marker_coords = marker_coords[marker_mask]
    
    # Move the object
    for y, x in active_coords:
        new_grid[y, x] = grid[np.clip(y - dy, 0, 63), np.clip(x - dx, 0, 63)] # This is wrong logic for translation
    
    # Correct Translation Logic:
    # 1. Clear old positions of active object and marker
    # 2. Set new positions
    
    # We need to know what was behind the object. In this game, it seems the background is color 3 or 4.
    # Based on deltas, when moving, the cells are replaced by values from a "background" pattern.
    # The observed transitions show that as the block moves, the area it leaves becomes color 3 (or whatever was there).
    # Let's assume the background is fixed and we just slide the object over it.
    
    # To simulate this perfectly, we would need the original background.
    # Since we don't have it, let's use the colors seen in the deltas.
    # Most movements replace the object with color 3.
    
    # For simplicity, since this is an ARC-AGI world model, let's try to find the movement patterns.
    # ACTION1: Up (-5 rows)
    # ACTION3: Left (-5 cols)
    # ACTION4: Right (+5 cols)
    
    # Re-evaluating based on deltas:
    # Initial: Object at r45-49, c24-28. Marker at r61-62, c14.
    # Action 3: Object -> r45-49, c19-23. Marker -> r61-62, c15.
    # Action 3 again: Object -> r45-49, c14-18? No, delta says r45c19... wait.
    # Looking closer at "changed cells":
    # ACTION3 (0->0): r45c24:12x5,3x5 ... means col 24 becomes 12 for 5 cells, then 3 for 5 cells.
    # This means it shifted LEFT by 5. The new values are [12,12,12,12,12] and [3,3,3,3,3].
    # So the object was at c24-28, now it's at c19-23, and c24-28 became color 3.
    
    # Let's implement a simple translation of the block (colors 12, 9) and marker (color 3 in rows 61-62).
    
    new_grid = grid.copy()
    obj_mask = (grid == 12) | (grid == 9)
    obj_coords = np.argwhere(obj_mask)
    if len(obj_coords) == 0: return new_grid
    
    marker_mask = (grid == 3) & (grid[:, :].ndim == 2) # just to be safe
    # Marker is specifically in r61-62
    marker_coords = np.argwhere((grid == 3) & ((np.arange(64)[:, None]) >= 61))
    
    dy, dx = 0, 0
    if action == 1: dy = -5
    elif action == 2: dy = 5
    elif action == 3: dx = -5
    elif action == 4: dx = 5
    else: return new_grid

    # Clear old object positions with color 3 (as seen in deltas)
    for y, x in obj_coords:
        new_grid[y, x] = 3
    
    # Move object
    for y, x in obj_coords:
        ny, nx = y + dy, x + dx
        if 0 <= ny < 64 and 0 <= nx < 64:
            new_grid[ny, nx] = grid[y, x]
            
    # Handle marker movement
    # The marker is a vertical line of color 3 at some column.
    # In ACTION3, it moves from c14 to c15? No, the delta says r61c14:3x1 becomes something else?
    # "r61c14:3x1" means cell (61, 14) is now value 3. But it was already 3.
    # Let's look at the marker again: r61c14:3x1, r62c14:3x1.
    # Then Action 3 -> r61c15:3x1, r62c15:3x1.
    # So the marker also shifts by 1 for every 5-pixel move of the block.
    
    m_dx = 0
    if action == 1: m_dx = 1 # Up moves marker right
    elif action == 2: m_dx = -1 # Down moves marker left
    elif action == 3: m_dx = 1 # Left moves marker right
    elif action == 4: m_dx = 1 # Right moves marker right... wait.
    
    # Looking at deltas:
    # ACTION3 (Left): Marker c14 -> c15
    # ACTION1 (Up): Marker c15 -> c16
    # ACTION1 (Up): Marker c16 -> c17
    # ACTION1 (Up): Marker c17 -> c18
    # ACTION4 (Right): Marker c18 -> c19
    # It seems the marker always moves RIGHT (+1) regardless of direction?
    # Let's check: Action 3(L)->+1, Action 1(U)->+1, Action 1(U)->+1, Action 1(U)->+1, Action 4(R)->+1.
    # Yes, every successful move shifts the marker one column to the right.
    
    m_coords = np.argwhere((grid == 3) & ((np.arange(64)[:, None]) >= 61))
    if len(m_coords) > 0:
        # Find current marker column (the most common x in rows 61-62 for color 3)
        curr_col = m_coords[:, 1].min() # Simplified
        # Clear old marker
        for y, x in m_coords:
            new_grid[y, x] = 11 # Background color in r61 is 11
        # Set new marker
        new_col = curr_col + 1
        if 0 <= new_col < 64:
            new_grid[61, new_col] = 3
            new_grid[62, new_col] = 3

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when the object reaches a target or marker matches.
    # Given no info, return False.
    return False