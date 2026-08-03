import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game state is a 64x64 integer grid.
    # Action 2 is likely 'Right' or similar movement of some entity.
    # Action 4 is likely 'Left' or similar movement of some entity.
    # Action 6 is a click.
    
    # Based on thes observed transitions, it seems like there are blocks of colors 2 and 9 moving.
    # same pattern of changes occur when ACTION 2 occurs.
    # laughtly, looking at thethought process (the own internal logic), we
    # identify that the same set of cells change from color 9 to 5 or vice versa.
    #
    # Let's try to implement a simple rule:
    # Action 2 moves a "cursor" or "active region" to the right.
    # Action 4 moves it to the left.
    # Action 3 is Up, Action 1 is Down? Or standard WASD/Arrow keys.
    # Usually in these games, 2=Down, 4=Up, 6=Left, 8=Right? No, let's check.
    # In this transition data, ACTION 2 causes things to move Right.
    # 
    # The entities being moved are blocks of size 5x5 (or variations).
    # Theentities are often replaced by background color 5.
    # The//// no_think
    # Actually, look at the delta: r8c14:5x5 r8c20:2x5 ...
    # 9x5 becomes 5x5 and something else becomes 2x5.
    # It seems like a block of color 2 is moving across the grid.
    # 
    # Let same as coordinates be (y, x) = (row, col).
    # Initial state has some structures.
    # Color 5 is likely the wall/background of the puzzle area.
    # Color 2 is the player/cursor.
    # Color 9 is the target/path.
    # Color 8 is the obstacle.
    # Action 2 moves the cursor right.
    # Action 4 moves it left.
    # 
    # Let's try to implement a movement rule for the cursor (color 2).
    
    new_grid = grid.copy()
    
    # Find the cursor position
    cursors = np.where(grid == 2)
    if len(cursors[0]) == 0:
        return new_grid
    
    # Cursor is usually a block. Let's find its bounding box.
    min_r, max_r = np.min(cursors[0]), np.max(cursors[0])
    min_c, max_c = np.min(cursors[1]), np.max(cursors[1])
    
    # Define direction based on action
    dr, dc = 0, 0
    if action == 2: # Right
        dc = 6
    elif action == 4: # Left
        dc = -6
    elif action == 3: # Up
        # Not seen in data but common
        dc = 0; dr = -6
    elif action == 1: # Down
        # not seen in data
        dc = 0; dr = 6
    
    # Move the cursor block
    # First, clear old position
    new_grid[min_r:max_r+1, min_c:max_c+1] = grid[min_r:max_r+1, min_c:max_c+1]
    # This logic is wrong. We need to actually move it.
    
    # Let's try a simpler approach: find all cells of color 2 and shift them.
    # The delta shows that blocks are moving by exactly 6 columns.
    # Shift coordinates of all color 2 pixels.
    coords = np.where(grid == 2)
    for r, c in zip(coords[0], coords[1]):
        # Check bounds
        nr, nc = r + dr, c + dc
        if 0 <= nr < 64 and 0 <= nc < 64:
            # In this specific game, the movement seems to be based on "tiles" of size 6x6 or similar.
            pass

    # Re-implementing based on observed deltas more carefully.
    # ACTION 2 moves things Right (dc=+6).
    # ACTION 4 moves things Left (dc=-6).
    # It looks like if a cursor block (color 2) moves into a cell of color 9, 
    # that cell becomes color 5 (background), and the old cell becomes color 2?
    # No, look at r8c14:5x5 r8c20:2x5. Color 5 replaces something, and color 2 appears.
    # This is basically shifting a pattern of color 2s across a field of color 9s.
    
    # Let's try: find all pixels of color 2, move them by (dr, dc).
    # If they land on color 9, it's allowed.
    # If they land on color 8, it's blocked.
    # If they land on color 0, it's outside.
    
    coords = np.where(grid == 2)
    if len(coords[0]) == 0: return new_grid
    
    # Clear current cursor positions
    for r, c in zip(coords[0], coords[1]):
        new_grid[r, c] = 5 # Assume background is 5
        
    # Place new cursor positions
    for r, c in zip(coords[0], coords[1]):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 64 and 0 <= nc < 64:
            # Check if target cell was color 9 or 5?
            # In the deltas, we see that cells are changing to 2 and 5.
            # The only other colors present in the path area are 9 and 5.
            new_grid[nr, nc] = 2
            
    return new_grid

def is_level_complete(grid):
    # Win state usually involves collecting all of some color or removing all of color 9.
    # Looking at the initial grid, there's a lot of color 9.
    # If no more color 9 pixels exist in the puzzle area, it might be complete.
    # Or look at the very bottom row (r63). It has a progress bar.
    # The progress bar is color 1.
    # If the progress bar fills up (color 1 reaches column 0), it's complete.
    return np.sum(grid == 1) >= 62 # Initial was 2x1 at col 62.

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for g50t is that all cells are the same color (uniform grid).
    """
    grid = np.array(grid)
    if grid.size == 0:
        return False
    return np.all(grid == grid[0, 0])
