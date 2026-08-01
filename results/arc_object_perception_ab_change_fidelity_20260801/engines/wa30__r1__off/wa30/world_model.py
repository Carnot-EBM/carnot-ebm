import numpy as np

def engine(grid, action, data):
    """
    World model for game 'wa30'.
    The grid contains objects of different colors.
    Action 6 is a click on specific coordinates own.
    Other actions are keyboard/directional movement.
    """
    # Copy the same grid to avoid modifying it in place
    next_grid = grid.copy()
    
    # Based on observed transitions, we can induce that Action 4 moves an object or changes state
    # The delta shows updates at r32c32, r33c32... and r35c40 etc.
    # This looks like a player character moving across a screen.
    # if action == 6: # Click
        # px, py = data['x'], data['y']
        #<|channel>thought
        # In this specific level layout, the "player" seems to be represented by color 0 (empty space)
        # and the "walls" are color 1 (background).
        # Let's assume the player is the only cell of color 0.
        # If they move, the old position becomes background (color 1) and new position becomes wall/object.
    
    # Since the provided observations are very limited and don't show explicit movements
    # but rather "deltas", we actually need to implement the logic that produces those deltas.
    
    # Looking at ACTION4: It shifts something from c32 -> c36 -> c40 -> c44.
    # This is a movement of +4 columns per call.
    # Look at ACTION1: Shifts something vertically.
    # {r28-31}c48 -> {r24-27}c48. Movement of -4 rows.
    # Look at ACTION3: Changes values in a block.
    # Look at ACTION5: Resets some values.
    
    # However, without knowing *what* object moves or how it interacts with others,
    # the most robust way for this specific ARC task is to identify the moving entity.
    # In the INITIAL GRID, there is a gap of color 0 at r32-35, c33.
    # Let's assume the player is the cell(s) of color 0.
    
    # For simplicity, since we must provide an executable engine:
    # We will treat Action 4 as Move Right, Action 2 as Move Left (implied), 
    # Action 1 as Move Up, Action 3 as Move Down (implied).
    
    # But wait, looking at the deltas again:
    # ACTION4: r32c32 becomes 1x4... then r32c36... then r32c40...
    # This means cells that were 0 are becoming 1, and cells that were 1 are becoming 0.
    # The "hole" (color 0) is moving right by 4 units each time.
    
    # Let's find the hole (color 0).
    holes = np.argwhere(grid == 0)
    if holes.size == 0:
        return next_grid

    # Assume the hole is a block. Find its bounding box.
    r_min, c_min = holes.min(axis=0)
    r_max, c_max = holes.max(axis=0)
    
    # Define movement vectors for actions
    # Based on observations: Action 4 moves hole RIGHT (+4 cols)
    # Action 1 moves hole UP (-4 rows)
    move_map = {
        4: (0, 4),  # Right
        2: (0, -4), # Left (inferred)
        1: (-4, 0), # Up
        3: (4, 0),  # Down (inferred)
    }
    
    if action in move_map:
        dr, dc = move_map[action]
        # Current hole area
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                if grid[r, c] == 0:
                    next_grid[r, c] = 1 # Fill old hole with background
        
        # New hole area
        nr_min, nc_min = r_min + dr, c_min + dc
        nr_max, nc_max = r_max + dr, c_max + dc
        
        # Boundary check
        if 0 <= nr_min < grid.shape[0] and 0 <= nr_max < grid.shape[0] and \
           0 <= nc_min < grid.shape[1] and 0 <= nc_max < grid.shape[1]:
            for r in range(nr_min, nr_max + 1):
                for c in range(nc_min, nc_max + 1):
                    next_grid[r, c] = 0
    
    return next_grid

def is_level_complete(grid):
    """
    The level is complete when the 'hole' reaches a target or collects something.
    Looking at ACTION4 (level 0->0), it doesn't win yet.
    Usually, winning involves reaching color 7 or similar.
    In INITIAL GRID, r63c57-63 are color 4.
    ACTION4 delta shows r63c56 becoming 4x1. This suggests interaction with the bottom row.
    """
    # A common win condition in these games is to clear all objects of a certain color
    # or reach a specific coordinate. Since we don't have a WIN STATE grid:
    # We check if any cell of color 0 has reached the same column as the goal area (color 4).
    holes = np.argwhere(grid == 0)
    if holes.size == 0: return False
    
    # Check if hole is near the bottom right goals
    for r, c in holes:
        if r >= 60 and c >= 50:
            return True
    return False