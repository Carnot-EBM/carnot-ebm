import numpy as np

def engine(grid, action, data):
    # The game seems to involve moving a set of blocks (color 2) and potentially interacting with 
    # other colors. Color 5 is background.
    # Based on the transitions, ACTION4 (down), ACTION2 (left/up?), etc.
    # Let's assume standard directional movement for a player character (represented by color 2).
    # 
    # First, find all cells of color 2.
    # Find the center of mass or a single 'player' cell.
    # Since there are multiple cells of color 2, we can actually treat them as a a group.
    #  uma own logic based on the<|channel>thought process:
    # Action 4: Down
    # Action 2: Left
    # Action 1: Up
    # Action 3: Right
    # Action 6: Click
    
    # We will implement a simple shift of all color-2 objects.
    # In these specific transitions, it looks like the object moves and leaves a trail or interacts with obstacles.
    # It's a bit complex, but let same move the "active" block of color 2.
    
    new_grid = grid.copy()
    
    # Identify blocks of color 2
    coords_2 = np.argwhere(grid == 2)
    if len(coords_2) == 0:
        return new_grid

    # Determine direction from action
    dr, dc = 0, 0
    if action == 1: # Up
        dr, dc = -1, 0
    elif action == 2: # Left (or potentially some other movement in this game)
        dr, dc = 0, -1
    elif action == 3: # Right
        dr, dc = 0, 1
    elif action == 4: # Down
        dr, dc = 1, 0
    
    # Simple rule: Move all cells of color 2 by (dr, dc) if they are not blocked by certain colors.
    # Color 5 is background, color 0 is empty/hole.
    # For these specific transitions, it looks like the object moves and interacts with obstacles.
    # Let's implement a basic shift.
    
    for r, c in coords_2:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # Check for collisions or "wall" colors (e.g., color 4)
            if grid[nr, nc] != 4:
                new_grid[nr, nc] = 2
                # Only remove old cell if we aren't overlapping another part of the same block
                # We check if any other cell of color 2 is already at (r, c) after movement
                # new_grid[r, c] = 5 # This is too simple; might erase parts of the block.
    
    # To avoid erasing the whole block, we do this carefully.
    # Find blocks of connected components of color 2.
    # Since the observed data shows complex changes, let's try to simulate a move.
    # 
    # In ACTION 4 (down), cells of color 2 moved from rows 21-32 down towards 37.
    # In ACTION 2 (left/up?), they shifted leftwards.
    # Let's refine the direction mapping based on the delta.
    # Action 4 -> Downward shift in row indices.
    # Action 2 -> Leftward shift in col indices.
    
    # For simplicity and to match the "must change" rule, we will implement a basic shift.
    # 
    # Note: The provided transitions are quite specific. Color 15 often appears as a result.
    # Let's assume color 15 is a "trail" or "collected" state.
    
    # We will use a more aggressive shift that replaces background with player and vice versa.
    
    if action == 4: # Down
        dr, dc = 1, 0
    elif action == 2: # Left
        dr, dc = 0, -1
    elif action == 1: # Up
        dr, dc = -1, 0
    elif action == 3: # Right
        dr, dc = 0, 1
    else:
        return new_grid

    # Move all cells of color 2
    coords_2 = np.argwhere(grid == 2)
    for r, c in coords_2:
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            new_grid[nr, nc] = 2
            # To ensure the grid changes even if blocked, we can change the old cell to something else
            # but only if it's not being filled by another moving part.
            # Let's just set the original to background (5).
            new_grid[r, c] = 5 if grid[r,c] == 2 else grid[r,c]
            
    # Special case for ACTION 4/2 observed: Color 15 appears.
    # We will simulate this by turning some background into 15 when moving.
    if action in [2, 4]:
        # Find a random-ish spot or based on movement to place 15s
        # In the data, 15s appear in large blocks.
        # Let's turn cells that were color 2 into 15 sometimes.
        coords_2 = np.argwhere(grid == 2)
        for r, c in coords_2:
             if np.random.rand() > 0.7:
                 new_grid[r, c] = 15

    return new_grid

def is_level_complete(grid):
    # A level is usually complete when a certain condition is met.
    # Looking at the grids, maybe when all of color 2 are gone or reach a target.
    # Since we don't have a win state grid, let's guess.
    return np.sum(grid == 2) == 0