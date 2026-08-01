import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # The object being moved is a 3x3 block of color 9 (or similar) that moves within a maze of walls (color 2/0/5).
    # Color 5 is background.
    # Color 9 is the player/object.
    # Color 0 is empty space / path.
    # Color 2 is wall.
    # Let's identify the current position of the player (the 3x3 block of color 9).
    
    h, w = grid.shape
    player_pos = None
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 9:
                player_pos = (r, c)
                break
        if player_pos:
            break
    
    if player_pos is None:
        return grid.copy()

    # The player is a 3x3 block. Find its top-left corner.
    pr, pc = player_pos
    # Ensure we have the top-left most cell of color 9.
    # find all cells of color 9
    coords = np.argwhere(grid == 9)
    if len(coords) == 0:
        return grid.copy()
    
    # We assume the player is always a 3x3 block.
    #<|channel>thought
    # In the observed transitions, the delta shows the 3x3 block of color 9 moving.
    # The object moves by 6 units? No, let's look at the deltas again.
    # ACTION2: r21c33 -> r27c33 (diff=6 rows).
    # ACTION3: r33c33 -> r33c27 (diff=6 cols).
    # ACTION4: r33c21 -> r33c27 (diff=6 cols).
    # It seems the movement step size is 6 pixels/units.
    
    step = 6
    dr, dc = 0, 0
    if action == 2: # Down
        dr = step
    elif action == 3: # Left
        dc = -step
    elif action == 4: # Right
        dc = step
    else:
        return grid.copy()

    new_pr, new_pc = pr + dr, pc + dc
    
    # Check for collisions with walls (color 2 or other non-path colors)
    # We need to check if the target area can accommodate a 3x3 block.
    # In this game, it looks like the player moves between "slots".
    # The slots are likely defined by the layout of color 0 and 2.
    # Let's see if we move only if the destination is 'empty' (color 0 or similar).
    # But in the deltas, some cells change from 9 to 0 and others from 0 to 9.
    # This suggests the path consists of color 0.
    
    # To simplify, let's just apply the movement if it stays within bounds.
    # Since we don't have full wall logic, we'll implement the basic shift.
    
    new_grid = grid.copy()
    
    # Find all current player cells
    player_cells = np.argwhere(grid == 9)
    for r, c in player_cells:
        new_grid[r, c] = 0 # Clear old position
        # Note: Some transitions show color 4 appearing briefly? 
        # "9x1,4x1,9x1" - this might be an animation frame or specific cell property.
        # We will stick to moving the block of 9s.
    
    # Place player at new position
    for r, c in player_cells:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            new_grid[nr, nc] = 9
            
    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly provided as a grid, but usually involves reaching a target.
    # In the observed data, there are no win states shown.
    # Based on common ARC patterns, we check if the object reached a certain area (e.g., color 14).
    # Let's look for any overlap between the player (color 9) and the goal (color 14).
    player_coords = np.argwhere(grid == 9)
    goal_coords = np.argwhere(grid == 14)
    if len(player_coords) == 0 or len(goal_coords) == 0:
        return False
    
    # Check if any player cell is near or on a goal cell
    for pr, pc in player_coords:
        for gr, gc in goal_coords:
            if abs(pr - gr) <= 2 and abs(pc - gc) <= 2:
                return True
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells in the grid are the same color (specifically, color 1).
    """
    grid = np.array(grid)
    return np.all(grid == 1)
