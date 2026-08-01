import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid is a 64x64 integer array
    # Action mapping based on observed transitions:
    # ACTION 1: Up, ACTION 2: Down, ACTION 3: Left, ACTION 4: Right, ACTION 5: Reset/Clear?
    # Based on the delta analysis, it seems like there's a "player" or "active object" (color 14)
    # and some other blocks (colors 4, 9, 2, 0).
    # The player (color 14) moves in directions specified by actions.
    # The same logic applies to any object that can be move own direction.
    #<|channel>thought
    # Let's refine the movement rules:
    # ACTION 1: Move everything shift-up
    # ACTION 2: Move everything shift-down
    # ACTION 3: Move left
    # ACTION 4: Move right
    # ACTION 5: Clear specific objects
    # Looking at the deltas:
    # ACTION 4 (Right): r32c32:1x4,14x3,0x1 -> r32c36:1x4,14x3,0x1 ...
    # This is a block of color 14 moving from col 32 to 36, then 36 to 40, etc.
    # It jumps by 4 columns each time.
    # The block of color 14 is 4 rows high (32-35) and 3 cols wide (32-34).
    # The block of color 0 is 4 rows high (32-35) and 1 col wide (35).
    # So it's a combined unit of width 4 (3+1).
    # ACTION 1 (Up): Moves blocks up.
    # ACTION 3 (Left): Moves blocks left.
    # ACTION 5: Seems to be a "clear" or "reset" action that removes some blocks.

    # Identify all movable objects (colors other than 1 and 7)
    # We assume colors 0, 2, 4, 9, 14 are the "game pieces"
    # Based on the observed transitions, movement is in increments of 4 pixels/units.
    # Let'<|channel>thought
    # Wait, looking at the deltas again:
    # ACTION 4 moves r32c32... to r32c36... which is +4 columns.
    # ACTION 1 moves r28c48... to r24c48... which is -4 rows.
    # This suggests a grid-based movement where each cell is actually a 4x4 block?
    # Or just that these specific objects move by 4 units.
    
    # Let's implement simple translation for any object not color 1 or 7.
    
    new_grid = grid.copy()
    
    if action == 1: # Up
        shift_r, shift_c = -4, 0
    elif action == 2: # Down
        shift_r, shift_c = 4, 0
    elif action == 3: # Left
        shift_r, shift_c = 0, -4
    elif action == 4: # Right
        shift_r, shift_c = 0, 4
    elif action == 5: # Clear/Reset (observed as removing some blocks)
        # In the observed transition, it removes obj1 (color 4 at 24,44).
        # Since we don't have a general rule, let's try to clear all non-background colors in the center.
        mask = (grid != 1) & (grid != 7)
        new_grid[mask] = 1
        return new_grid
    else:
        return new_grid

    # Movement logic: move all pixels that are not background (1) or boundary (7)
    # and handle collisions by simply shifting them.
    movable_mask = (grid != 1) & (grid != 7)
    coords = np.argwhere(movable_mask)
    
    # To avoid overwriting while moving, we start with a background grid
    # But wait, the background is color 1.
    temp_grid = np.full_like(grid, 1)
    # Keep boundaries
    temp_grid[grid == 7] = 7
    
    for r, c in coords:
        nr, nc = r + shift_r, c + shift_c
        if 0 <= nr < 64 and 0 <= nc < 64:
            temp_grid[nr, nc] = grid[r, c]
        else:
            # If it goes off board, maybe it wraps or stays? Let's assume it stays.
            pass
            
    return temp_grid

def is_level_complete(grid):
    # Win state usually involves reaching a certain configuration.
    # In this game, looking at the INITIAL GRID, there are blocks of colors 4, 9, 2, 0, 14.
    # The win state isn't explicitly given, but often it's when all "player" pieces reach a goal.
    # Based on ACTION 4 (Right), some cells change to 4 at r63c56.
    # This suggests that moving objects to the bottom-right might be the goal.
    # Let's check if any color 4 block has reached the boundary area (row 63).
    return np.any((grid == 4) & (np.arange(64)[:, None] == 63))

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells are the same color (excluding background 0).
    """
    grid = np.array(grid)
    non_zero = grid[grid != 0]
    if non_zero.size == 0:
        return False
    return np.all(non_zero == non_zero[0])
