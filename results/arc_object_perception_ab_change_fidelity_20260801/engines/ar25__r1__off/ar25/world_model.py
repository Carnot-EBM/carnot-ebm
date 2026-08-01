import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is 64x64. Colors are [0, 4, 5, 9, 10, 11].
    # Action 2: Move Down
    # Action 3: Move Left (based on the observed transitions)
    # The game seems to be a puzzle where blocks of colors move and change based on certain rules.
    # Based on the same pattern in ACTION2's deltas, it's a<|channel>thought
    # that the player controls an object or set of objects that they can move.
    # Let's implement a simple movement logic for own-state tracking.
    
    new_grid = grid.copy()
    
    if action == 2: # Down
        # Shift specific patterns down by 1 row.
        # This looks like a "sliding" mechanism.
        # We need to identify the moving parts.
        # In ACTION2, we see cells changing from color 9 to something else and vice versa.
        #
        # Looking at the delta: r18c3:9x9, r18c51:9x9... r27c9:5x3, r27c51:4x3
        # It appears that some regions are being replaced by background color 9.
        # And new regions are appearing as colors 5 and 4.
        #
        # Since I cannot deduce the exact physics engine without more data, 
        # but must provide a functional world model based on these transitions,
        # I will map the observed movements.
        
        # The pattern in Action 2 is consistent: it shifts blocks of colors (5, 4) downwards.
        # For each cell (r, c), if it's not background (9) or wall (10), check if it can move down.
        # Let's try shifting all non-background/non-wall pixels down by one.
        
        # Mask for movable objects (colors other than 9 and 10).
        movable_mask = (grid != 9) & (grid != 10)
        
        # Shift mask down
        shifted_mask = np.roll(movable_mask, shift=1, axis=0)
        
        # We need to handle boundaries and collisions.
        # In this game, walls (10) seem to be static vertical strips.
        # This means movement is likely constrained within columns.
        
        # To simulate ACTION2 (Down):
        # 1. Identify current positions of "objects" (not color 9 or 10).
        # 2. Move them down by 1 row.
        # 3. Restore the original cells to background (color 9) unless they are now occupied.
        # 4. Handle collision with walls (10) - though in own data walls are at col 30-33.
        
        # Simple implementation:
        # Find all blocks that aren't 9 or 10.
        # Move them down.
        
        # Create a copy for the new state.
        temp_grid = grid.copy()
        
        # Set movable pixels back to background (9)
        temp_grid[movable_mask] = 9
        
        # Place shifted pixels into temp_grid
        for r in range(63, -1, -1):
            for c in range(64):
                if movable_mask[r, c]:
                    if r + 1 < 64 and temp_grid[r+1, c] == 9:
                        temp_grid[r+1, c] = grid[r, c]
                    else:
                        # If it can't move, it stays put.
                        temp_grid[r, c] = grid[r, c]
        
        return temp_grid

    if action == 3: # Left
        # Similarly shift non-background/non-wall pixels left by one.
        temp_grid = grid.copy()
        movable_mask = (grid != 9) & (grid != 10)
        temp_grid[movable_mask] = 9
        
        for r in range(64):
            for c in range(64):
                if movable_mask[r, c]:
                    if c - 1 >= 0 and temp_grid[r, c-1] == 9:
                        temp_grid[r, c-1] = grid[r, c]
                    else:
                        temp_grid[r, c] = grid[r, c]
        
        return temp_grid

    return new_grid

def is_level_complete(grid):
    # The win state isn't explicitly provided as a full grid, but usually involves
    # moving objects to specific target locations or clearing the board.
    # For now, we return False unless a clear condition is met.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is based on the presence of a specific pattern or 
    the same color filling the same area.
    """
    grid = np.array(grid)
    
    # The win condition for ar25 is typically that the grid 
    # contains only one color (excluding background 0).
    # We check if all non-zero elements are the same color.
    non_zero = grid[grid != 0]
    if non_zero.size == 0:
        return False
    
    return np.all(non_zero == non_zero[0])
