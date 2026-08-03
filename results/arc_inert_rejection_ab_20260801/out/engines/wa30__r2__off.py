import numpy as np

import numpy as np

def engine(grid, action, data):
    # This is a world model for wa30.
    # Based on the observed transitions, it seems to be a puzzle game where blocks of colors are moved or modified.
    # The grid contains various colored regions.
    # Action 4 is likely 'right' movement of some object.
    #<|channel>thought
    # Action 1 is 'up'.
    # Action 3 is 'down'.
    # Action 5 is 'left'.
    # Action 1: Up, Action 3: Down, Action 5: Left, Action 4: Right? No, wait.
    # Let's look at the coordinates.
    # ACTION 4 moves something from c32->c36->c40->c44. (Right)
    # ACTION 1 moves something from r28-31 -> r24-27. (Up)
    # ACTION 3 moves something from r24-27 -> r28-31. (Down)
    # ACTION 5 moves something back. (Left)
    # It looks like there is a player/cursor block that can move in 4 directions.
    # In this specific case, the "player" is a 4x4 block of color 14 (or similar).
    # Find the 4x4 block of color 14 and move it.
    
    new_grid = grid.copy()
    
    # Identify the moving object (the 4x4 block of color 14)
    # We search for the top-left corner of a 4x4 block of color 14.
    obj_pos = None
    for r in range(61):
        for c in range(61):
            if np.all(grid[r:r+4, c:c+4] == 14):
                obj_pos = (r, c)
                break
        if obj_pos: break

    if obj_pos is None:
        return new_grid

    r, c = obj_pos
    dr, dc = 0, 0
    if action == 1: # Up
        dr = -4
    elif action == 3: # Down
        dr = 4
    elif action == 5: # Left
        dc = -4
    elif action == 4: # Right
        dc = 4
    else:
        return new_grid

    nr, nc = r + dr, c + dc
    
    # Check boundaries
    if nr < 0 or nr > 60 or nc < 0 or nc > 60:
        return new_grid

    # Simple movement: clear old position and set new position
    # In the observed data, it seems some cells are replaced by color 1 (background).
    # Let's try to implement a basic move.
    
    # Clear current block
    new_grid[r:r+4, c:c+4] = 1
    # Set new block
    new_grid[nr:nr+4, nc:nc+4] = 14
    
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly given, but usually involves reaching a target area.
    # Looking at the initial grid, there's a region of color 4 at the bottom right (r63, c57-63).
    # If the player block reaches that area, it might be complete.
    # However, without a win state example, we guess based on common ARC patterns.
    # For now, return False unless a specific condition is met.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for 'wa30' is that all cells are the same color (excluding background).
    """
    # Convert to numpy array for easier manipulation
    grid_np = np.array(grid)
    
    # Find all non-zero (non-background) colors
    non_zero_colors = grid_np[grid_np != 0]
    
    # If there are no non-zero colors, it's not a win state
    if non_zero_colors.size == 0:
        return False
    
    # Check if all non-zero colors are the same
    return np.all(non_zero_colors == non_zero_colors[0])
