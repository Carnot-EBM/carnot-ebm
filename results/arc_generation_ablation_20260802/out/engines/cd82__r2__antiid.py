import numpy as np

def engine(grid, action, data):
    # The game seems to involve moving a set of pixels (color 2) of a certain shape
    # based on directional actions.
    # ACTION4: Right/Down?
    # ACTION2: Up/Left?
    # ACTION2 repeated suggests movement of the same object.
    # Based on the transitions, color 2 blocks move across the grid.
    # Let's implement a simple translation of all cells of color 2.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right/Down
        # Shift color 2 blocks right and down slightly
        # shift_x = 1
        # shift_y = 1
        # shift_z = 0
        #<|channel>thought
        # Looking at the delta, r21c39:2x1 means cell (21, 39) becomes 2.
        # In INITIAL GRID, there are no 2s in that area.
        # Color 2 exists in rows 24-32.
        # Action 4 moves them from [24-32, 25] and [24-32, 37] to new positions.
        # The observed deltas for Action 4 are complex.
        # It looks like it shifts the "object" composed of color 2 pixels.
        # shift_y = -1, shift_x = 1? No.
        # Let's try a simpler rule: translate all color 2 pixels by some vector.
        
        # Find all coordinates of color 2
        coords = np.argwhere(grid == 2)
        if len(coords) > 0:
            # For action 4, let' same as ACTION4 transition
            # Shift right and up slightly
            dy, dx = -1, 1
            for y, x in coords:
                new_grid[y, x] = grid[y, x] if grid[y, x] != 2 else 5 # Clear old position
                # Note: this is not quite correct because we need to clear first
                pass
    
    # Since the exact movement logic is very complex (and potentially involves collisions),
    # we can simulate the specific transitions provided.
    # However, the prompt asks for general rules.
    #
    # Looking at the data again:
    # Initial Grid: Color 2 blocks are at r24-32, c26 and c37.
    # Action 4: Moves them towards the center/right.
    # Action 2: Moves them further.
    # Let's implement a translation based on the action.
    
    res = grid.copy()
    coords = np.argwhere(grid == 2)
    
    if action == 4:
        # Move Right-Up?
        dy, dx = -2, 2
    elif action == 2:
        # Move Left-Down?
        dy, dx = 2, -2
    else:
        return res

    # To avoid overwriting while moving, we create a mask
    mask_2 = (grid == 2)
    res[mask_2] = 5 # Replace old color 2 with background color 5
    
    for y, x in coords:
        ny, nx = y + dy, x + dx
        if 0 <= ny < res.shape[0] and 0 <= nx < res.shape[1]:
            res[ny, nx] = 2
            
    return res

def is_level_complete(grid):
    # Usually complete when an object reaches a target or all of one color are gone.
    # In this game, it might be when color 2 pixels reach a certain area.
    # There's no win state provided, so let's guess based on common ARC patterns.
    return np.sum(grid == 15) == 0