import numpy as np

import numpy as np

def engine(grid, action, data):
    if action != 6:
        return grid
    
    # The game is a puzzle where clicking on specific coordinates moves 
    # certain blocks or changes colors based on some logic.
    # Based on the observed transitions, ACTION6 (click) at (61, 33) 
    # seems to trigger a shift in the board state.
    # Specifically, it looks like the same click coordinate (x=61, y=33) 
    # is repeated multiple times.
    # Each time it's clicked, the same set of cells change values.
    #<|channel>thought
    # Looking at the delta for the first ACTION6:
    # r0c63:4x1 -> cell (0, 63) becomes color 4
    # r1c32:3x4 -> cells (1, 32-35) become color 3
    # ... and so on.
    # It appears that the "empty" space (color 0) is being filled by the "wall" (color 3).
    # This suggests a movement of the wall boundary.
    # The wall (color 3) expands into the empty area (color 0).
    #
    # Let's refine this:
    # In the first transition, the x-boundary of color 3 moves from 32 to 36? No.
    # Wait, looking at the deltas:
    # Transition 1: r1c32:3x4 means columns 32, 33, 34, 35 are now color 3.
    # Transition 2: r1c36:3x4 means columns 36, 37, 38, 39 are now color 3.
    # This is clearly a horizontal expansion of the color 3 region.
    # Each click shifts the boundary by 4 units.
    #
    # Now let's look at other objects.
    # Color 9 blocks (obj3, obj8) are at x=60..63.
    # Color 4 and 11 blocks (obj9, obj10) are at x=46..51.
    # After the first click, they move to x=42..47?
    # Delta 1: r44c44:4x2,11x2,0x4 -> cells (44, 44-45) = 4, (44, 46-47) = 11, (44, 48-51) = 0.
    # Original was (44, 46-47)=4, (44, 48-49)=11.
    # So it shifted left by 2 pixels.
    # Transition 2: r44c40:4x2,11x2,0x4 -> shift left by another 4 pixels? No, from 44 to 40 is a shift of 4.
    #
    # The pattern:
    # Click shifts color 3 boundary right by 4.
    # Click shifts other objects (color 4, 11, etc.) left by 4.
    #
    # Let's check if this is general.
    # Initial wall boundary at x=32.
    # After click 1: boundary moves to 36.
    # After click 2: boundary moves to 40.
    #
    # Other objects:
    # Color 4/11 block initial x=[46, 49] and [50, 51].
    # After click 1: x becomes [44, 47] and [48, 49]? Wait, delta says r44c44:4x2,11x2... so x=44..45 is 4, x=46..47 is 11.
    # That's a shift of -2.
    # After click 2: r44c40:4x2,11x2... so x=40..41 is 4, x=42..43 is 11.
    # That's another shift of -4 from the previous state (44 -> 40).
    #
    # So: Wall shifts +4, Objects shift -4.
    #
    # Let's check if this continues until they meet or something.
    # The win condition seems to be when the wall reaches some point or objects are pushed.
    # In the "GRID BEFORE THE COMPLETING ACTION", the wall boundary is at x=40.
    # One more click moves it to 44.
    #
    # Now let's implement the engine based on these rules.
    
    new_grid = grid.copy()
    
    # Find current wall boundary (color 3)
    # We assume color 3 is a block starting from col 0.
    wall_boundary = 0
    for c in range(64):
        if np.all(grid[1:64, c] == 3): # This might be too strict
            wall_boundary = c + 1
        else:
            break
            
    # Actually, looking at the data, the wall is not perfectly rectangular.
    # But for most rows r1-r27, it's columns 0..31.
    # Let's find the average boundary of color 3.
    rows_to_check = range(1, 28)
    boundaries = []
    for r in rows_to_check:
        count = 0
        while count < 64 and grid[r, count] == 3:
            count += 1
        boundaries.append(count)
    avg_boundary = int(np.mean(boundaries)) if boundaries else 0
    
    # Shift wall right by 4
    for r in range(1, 64):
        for c in range(avg_boundary, min(avg_boundary + 4, 64)):
            # Only replace empty space (color 0) or other colors?
            # The delta shows it replaces 0s.
            if new_grid[r, c] == 0:
                new_grid[r, c] = 3

    # Shift objects left by 4
    # We need to identify "objects" (non-wall, non-background colors).
    # Background is 7 (top row), 0 (empty). Wall is 3.
    # Objects are 4, 5, 9, 11.
    obj_colors = {4, 5, 9, 11}
    
    # To shift objects, we find all cells of obj_colors and move them.
    # This is tricky because they might overlap.
    # Let's use a mask.
    mask = np.isin(grid, list(obj_colors))
    # Remove the shifted area from the grid first
    new_grid[mask] = 0 # This is wrong; should only remove if not wall
    # Wait, let's just iterate over the grid and move pixels.
    
    # Correct approach for shifting objects:
    # 1. Identify all object pixels.
    # 2. Create a new empty grid for objects.
    # 3. Place each pixel at (r, c-4).
    # 4. Merge with the rest of the grid.
    
    # But wait, the delta shows that color 3 also replaces some things?
    # No, it seems to be a simple translation.
    
    # Let's try a simpler rule:
    # If action == 6 and data == {'x': 61, 'y': 33}:
    #   Shift everything in columns [avg_boundary, 64) left by 4?
    #   And fill the gap with color 3?
    
    # Looking at Delta 1 again:
    # r1c32:3x4 -> cols 32,33,34,35 become 3.
    # r32c48:0x4 -> cols 48,49,50,51 become 0.
    # This looks like:
    # For rows where there was an object at x, it's now at x-4.
    # And the area from wall_boundary to wall_boundary + 4 becomes color 3.
    
    # Let's implement this "shift window" logic.
    wall_b = avg_boundary
    for r in range(64):
        # Shift pixels from [wall_b+4, 64) to [wall_b, 60)
        row_segment = grid[r, wall_b+4:].copy()
        new_grid[r, wall_b : wall_b + len(row_segment)] = row_segment
        # Fill the new gap [wall_b, wall_b+4) with color 3 (if not top row)
        if r > 0:
            for c in range(wall_b, min(wall_b + 4, 64)):
                new_grid[r, c] = 3
        # Clear the end of the row
        if wall_b + len(row_segment) < 64:
            # The delta shows some cells become 0.
            pass # handled by slicing if we are careful
            
    # Special case for top row (color 7)
    # Delta 1: r0c63:4x1 -> cell (0, 63) becomes 4? No, that's a typo in my reading.
    # "r0c63:4x1" means at row 0, col 63, value is 4, count is 1.
    # This means the object at the very edge shifted.
    
    return new_grid

def is_level_complete(grid):
    # Based on the win transition, the level is complete when the wall boundary reaches x=44.
    # Let's check the average boundary of color 3.
    rows_to_check = range(1, 28)
    boundaries = []
    for r in rows_to_check:
        count = 0
        while count < 64 and grid[r, count] == 3:
            count += 1
        boundaries.append(count)
    avg_boundary = int(np.mean(boundaries)) if boundaries else 0
    return avg_boundary >= 44

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state for game vc33.
    The win condition is that all cells in the grid are the same color.
    """
    grid = np.array(grid)
    if grid.size == 0:
        return False
    
    # Check if all elements are the same
    return np.all(grid == grid[0, 0])
