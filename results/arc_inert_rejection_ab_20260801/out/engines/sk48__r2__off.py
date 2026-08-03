import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the player controls an object (represented by color 6)
    # and moves it around the grid. Color 5 is background/wall.
    # Action 1: Move Up
    # Action 2: Move Down
    # Action 3: Move Left
    # Action 4: Move Right
    # Action 5: Not used in observed transitions
    # Action 6: Click - not used in observed transitions
    # Action 7: Not used in observed transitions
    
    # Find current position of the player character (color 6)
    # Note: In the initial grid, there are multiple cells of color 6.
    # These might be represent different parts of the same "object" or a "player".
    # Let's assume the<|channel>thought
    # find all coordinates of color 6.
    coords = np.argwhere(grid == 6)
    if len(coords) == 0:
        return grid
    
    # We need to determine which part of the object is the 'center' or how it moves.
    # Looking at the delta, ACTION1 (Up) changes r18c11...r29c11.
    # The object seems to be a block of color 6 and some other colors.
    # coords_6 = np.argwhere(grid == 6)
    
    # Simple movement logic based on the provided deltas:
    # ACTION 1: Up, ACTION 2: Down, ACTION 3: Left, ACTION 4: Right.
    # This doesn't match standard WASD/Arrow keys usually mapped to 1-4 in ARC.
    # However, looking at the transitions:
    # Action 1 shifted the block from rows 24-29 up to 18-23.
    # Action 2 shifted the block from rows 18-23 down to 24-29.
    # Then another Action 2 shifted it further down to 30-35.
    # Action 3 changed columns 36-42.
    # Action 4 changed columns 23, 29, 35, 41.
    
    # Let's refine the action mapping:
    # Action 1: Move Up (y - delta)
    # Action 2: Move Down (y + delta)
    # Action 3: Move Left (x - delta)
    # Action 4: Move Right (x + delta)
    
    # The "object" is a complex shape consisting of color 6 and others.
    # We need to identify all cells that are part of this moving object.
    # In the initial grid, there's a cluster around r24c11.
    # It seems everything not equal to background colors (like 5 or 4) might be part of it?
    # No, looking at the INITIAL GRID, color 4 is also present in large blocks.
    # Backgrounds seem to be 5 (top/bottom/sides) and 4 (central area).
    # Moving object consists of colors {0, 1, 2, 3, 6, 8, 9, 14}.
    
    obj_colors = {0, 1, 2, 3, 6, 8, 9, 14}
    mask = np.isin(grid, list(obj_colors))
    
    # Find the bounding box of the moving object.
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return grid
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Define movement offsets based on observed transitions.
    # Action 1 shifted by ~6 rows up.
    # Action 2 shifted by ~6 rows down.
    # Action 3 shifted a block horizontally.
    # Action 4 seems to interact with specific columns or move something small.
    
    # Let's try a simpler approach: shift all non-background pixels.
    new_grid = grid.copy()
    dy, dx = 0, 0
    if action == 1: dy = -6
    elif action == 2: dy = 6
    elif action == 3: dx = -6 # Guessing offset
    elif action == 4: dx = 6  # Guessing offset

    # This is too simple. The deltas show complex changes (toggling colors).
    # Looking at ACTION 4 again: it changes only a few cells. It might be "interacting" with the environment.
    # ACTION 3 changed a whole block of color 8 and 4.
    
    # Given the complexity and limited data, let's implement basic movement for the object.
    # We will treat any cell not in {4, 5} as part of the moving object.
    
    obj_mask = np.isin(grid, [0, 1, 2, 3, 6, 8, 9, 14])
    coords = np.argwhere(obj_mask)
    if len(coords) == 0: return grid
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # The background seems to be color 4 in the center and 5 on the edges.
    # When the object moves, it leaves behind the background color that was there.
    bg_color = grid[y_min-1, x_min] if y_min > 0 else 5
    if bg_color not in [4, 5]: bg_color = 4

    new_grid = grid.copy()
    
    # Shift logic
    shift_y, shift_x = 0, 0
    if action == 1: shift_y = -6
    elif action == 2: shift_y = 6
    elif action == 3: shift_x = -6
    elif action == 4: shift_x = 6
    
    if shift_y == 0 and shift_x == 0:
        return new_grid

    # To move the object:
    # 1. Fill current position with background
    for r, c in coords:
        new_grid[r, c] = bg_color
        
    # 2. Place object at new position
    for r, c in coords:
        nr, nc = r + shift_y, c + shift_x
        if 0 <= nr < 64 and 0 <= nc < 64:
            new_grid[nr, nc] = grid[r, c]
            
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing objects.
    # No win state provided, so we'll return False unless a specific condition is met.
    # Looking at ACTION 4 (level 0->0), it doesn't seem to trigger completion.
    return False