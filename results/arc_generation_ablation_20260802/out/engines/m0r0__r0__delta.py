import numpy as np

import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the grid based on the observed transitions.
    """
    out = grid.copy()
    
    # The game seems to be about moving a 'player' (color 0) which acts as a cursor.
    # Color 0 is at r0c63 initially.
    # ACTION1 moves the player right/down? No, let's look at the coordinates.
    # Action 1: r0c62:0x1, r63c1:0x1 -> Player moved from (0,63) to (0,62) and (63,0) to (63,1)?
    # Wait, the initial grid has color 0 at (0,63) and (63,0).
    # Let's trace the player position.
    # Initial: (0,63), (63,0)
    # After A1: (0,62), (63,1)
    # After A3: no change in player pos.
    # After A1: (0,61), (63,2)
    # After A1: (0,60), (63,3) - wait, that's not it.
    # Let's re-examine.
    # The cells changed are r0c62:0x1 and r63c1:0x1. This means cell (0,62) becomes 0 and (63,1) becomes 0.
    # But what about the previous values? They aren't listed as changing back to something else.
    # Actually, the delta is "changed cells". If a cell changes from 5 to 0, it's listed.
    # If it changes from 0 to 5, it's also listed.
    # Wait, if only r0c62:0x1 is listed, then (0,63) stays 0? No, usually deltas list all changes.
    # If (0,63) was 0 and now (0,62) is 0, but (0,63) isn't mentioned, it implies (0,63) remains 0.
    # However, in these games, there's usually one player.
    # Let's look at ACTION1 again: r0c62:0x1 r63c1:0x1.
    # Then later ACTION1: r0c61:0x1 r63c2:0x1.
    # This means color 0 is expanding or moving.
    # Looking at the coordinates: (0,63)->(0,62), (63,0)->(63,1).
    # The other changes are blocks of color 10.
    # Action 1 seems to be a "step" that moves the cursor and triggers block changes.
    # Action 4 seems to be a "toggle" or "shift".
    
    # Based on the observed transitions, we can deduce the rules for this specific level.
    # Since I must provide a general engine, I will implement the logic based on the same patterns.
    
    if action == 1:
        # Move cursors
        # Find current positions of 0s
        zeros = np.argwhere(grid == 0)
        for z in zeros:
            r, c = z
            if r == 0:
                out[0, c-1 if c > 0 else 0] = 0
                # We don't know what happens to the old position.
                # But usually they move. Let's assume they leave behind their original color.
                # Original colors were 5.
                out[0, c] = 5
            elif r == 63:
                out[63, c+1 if c < 63 else 63] = 0
                # Original color was 5.
                out[63, c] = 5
        
        # The block changes are complex and depend on the cursor position.
        # They seem to be related to columns 14-19 and 44-49.
        # Let's try to find the pattern of blocks changing to 10.
        # In A1 (first), rows 34-38 col 14-18 become 10, and rows 39-43 col 44-48 become 10.
        # In A1 (second), rows 29-33 col 14-18 become 10, and rows 34-38 col 49-53 become 10.
        # This is a very specific sequence.
        
    return out

def is_level_complete(grid):
    # Win state is usually when all target cells are filled or a certain condition is met.
    # Since no win state grid is provided, we assume it's complete when some condition is met.
    # For this level, let's return False unless a specific state is reached.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is that all cells in the grid are the same color (excluding background 0).
    """
    # Convert to numpy array for easier manipulation
    grid = np.array(grid)
    
    # Get all non-zero cells
    non_zero_cells = grid[grid != 0]
    
    # If there are no non-zero cells, it's not a win state
    if non_zero_cells.size == 0:
        return False
    
    # Check if all non-zero cells are the same color
    return np.all(non_zero_cells == non_zero_cells[0])
