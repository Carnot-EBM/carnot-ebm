import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is a 64x64 integer grid.
    # Action mapping based on observed transitions:
    # ACTION4: Click? No, the deltas show it's changing blocks of colors.
    # ACTION3: Right/Down movement or modification.
    # ACTION2: Up/Left movement or modification.
    # ACTION3 again: modifying specific regions.
    # Looking at the observed transitions, they seem to be related to filling or clearing areas.
    # However, the provided observations are not very detailed about own player position.
    # The "changed cells" describe modifications to specific rectangles or 3x3 blocks.
    # The same coordinates (r30, r31, r32) and then (r33, r34, r35) shift.
    # ACTION2 shifts from rows 30-32 to 33-35.
    # ACTION3 shifts columns 21 to 18 to 15.
    # ACTION4 seems to actually toggle or modify values in place.
    # This looks like a cursor-based game where you move a 3x3 block of color 1.
    # action 2: Down (rows +3)
    # action 3: Left (cols -3)
    # action 4: Toggle/Fill (changes current 3x3 area)
    # Let's refine this based on the<|channel>thought process.
    
    # We need to track the cursor position. Since it's not explicitly in the grid,
    # we can infer it from the grid state. The only way to know where the 'cursor' is,
    # if it's not stored as a unique value, is to look for the pattern of color 1.
    # But wait, the grid itself contains colors 0, 1, 2, 4, 5, 14, 15.
    # In the INITIAL GRID, there are regions of color 1 and 14.
    # Looking at r30c18:1x3,14x3... the cursor might be the 3x3 block of color 1.
    
    # Find the top-left of the 3x3 block of color 1.
    # Note: There might be multiple blocks of color 1. We assume the "active" one is moving.
    # However, since we must return a new grid, let's try to find any 3x3 block of color 1.
    
    coords = np.argwhere(grid == 1)
    if coords.size == 0:
        return grid # No cursor found
    
    # To handle potential noise, just take the first occurrence or center of mass.
    # The transitions show movement in steps of 3.
    # Let's identify the current 3x3 block of color 1.
    # Since it moves as a unit, we can use the minimum row/col.
    r_min, c_min = coords[0][0], coords[0][1]
    # But wait, if there are other color 1 cells, this fails.
    # Let's look for the most compact 3x3 cluster.
    
    # Based on the observed deltas:
    # ACTION4 (level 0->0): r30c18:1x3,14x3 -> This looks like replacing 1s with 14s?
    # ACTION2 (level 0->0): r30-32 (color 1) becomes something else, and r33-35 (color 14) becomes color 1.
    # This is definitely a moving 3x3 brush of color 1.
    
    # Action mapping:
    # action 2: Down (y + 3)
    # action 3: Left (x - 3)
    # action 4: Fill/Toggle (changes colors in current area)
    
    # Find top-left of the only 3x3 block of color 1.
    # We can find all pixels of color 1 and take the bounding box or min.
    # But we must be careful about existing color 1s.
    # In the initial grid, there are blocks of color 1 at rows 21-25 etc.
    # The "active" cursor is likely the one that moves.
    # Let's assume the active cursor is the most recently changed block of color 1.
    # Since we don't have state, let's look for any 3x3 block of color 1 that isn't part of a larger structure.
    
    # Actually, looking at ACTION2: it changes r30c21:1x3... to something else AND r33c21:14x3... to 1x3.
    # This means the brush *moves* from (30, 21) to (33, 21).
    # So the brush is always color 1.
    
    # To implement this simply:
    # 1. Find the current 3x3 block of color 1.
    # 2. Move its top-left corner based on action.
    # 3. Update the same area it left back to some default (maybe 14 or 15?).
    # 4. Set the new area to color 1.
    
    # But wait, what does ACTION4 do? It changes 1s to 14s and vice versa.
    # Let's assume the cursor is just a 3x3 window.
    
    # Since we don't know the starting position perfectly, let's find all 3x3 blocks of color 1.
    # Given the constraints and observed data, the most likely implementation:
    
    cursor_r, cursor_c = -1, -1
    # Search for a 3x3 block of color 1.
    for r in range(62):
        for c in range(62):
            if np.all(grid[r:r+3, c:c+3] == 1):
                cursor_r, cursor_c = r, c
                break
        if cursor_r != -1: break
    
    if cursor_r == -1: return grid

    new_grid = grid.copy()
    
    if action == 2: # Down
        # Restore old pos (approximate based on deltas)
        new_grid[cursor_r:cursor_r+3, cursor_c:cursor_c+3] = 14 
        # Move down
        nr, nc = cursor_r + 3, cursor_c
        if nr <= 61:
            new_grid[nr:nr+3, nc:nc+3] = 1
    elif action == 3: # Left
        new_grid[cursor_r:cursor_r+3, cursor_c:cursor_c+3] = 14
        nr, nc = cursor_r, cursor_c - 3
        if nc >= 0:
            new_grid[nr:nr+3, nc:nc+3] = 1
    elif action == 4: # Toggle/Fill
        # Based on ACTION4 delta: r30c18:1x3,14x3 -> it seems to flip 1s and 14s in the area.
        area = new_grid[cursor_r:cursor_r+3, cursor_c:cursor_c+3]
        mask1 = (area == 1)
        mask14 = (area == 14)
        area[mask1] = 14
        area[mask14] = 1
        new_grid[cursor_r:cursor_r+3, cursor_c:cursor_c+3] = area

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a certain color is gone or pattern reached.
    # Given no data, return False.
    return False

def is_level_complete(grid):
    # The same color (non-zero) is present in all rows and columns.
    # This is a<|channel>thought-process-//no_think
    # Simplified: Check if any non-zero value is present in all rows and columns.
    # Actually, the most common win state for these grids is a full grid or a specific pattern.
    # For a general rule: check if there are no zeros.
    # But the prompt asks for a specific game 'ka59'.
    # In ARC-AGI, 'complete' usually means the target pattern is achieved.
    # Since I cannot see the grid, I will implement a common win condition:
    # No zeros in the grid (completely filled).
    return not (0 in grid)
