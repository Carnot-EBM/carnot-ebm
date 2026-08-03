import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next state of the grid based on the observed transitions.
    The game involves a character (color 0) and obstacles/walls (color 10).
    ACTION4 is a move right.
    """
    # Create a copy to avoid modifying the original grid
    next_grid = grid.copy()
    
    # Find the position of the player character (color 0)
    # The initial grid shows color 0 in blocks. Let's assume the player is represented by the same logic
    # as movement own-cells.
    
    # In this specific level layout, it seems like ACTION4 moves the 'empty' spaces (color 0)
    # effectively shifting the pattern of zeros.
    
    # Find all cells that are currently color 0
    zeros = np.argwhere(grid == 0)
    if len(zeros) == 0:
        return next_grid
    
    # For each zero cell, if the same column index + 1 is not a wall (color 10), 
    # we shift the zero to the right.
    # This is essentially moving the "gap" or "player" to the right.
    # 
    # We find the gaps (color 0)
    # gap_mask = (grid == 0)
    # Shift the mask to<|channel>thought
    # shifted_mask = np.roll(gap_mask, 1, axis=1)
    #
    # Looking at the delta transitions, they are very consistent:
    # r14c11:10x3 means row 14, col 11 becomes color 10.
    # The gap at c11 was likely replaced by a wall, and a new gap opens up at c14.
    #
    # Let's refine the movement logic based on the observed ACTION4 deltas.
    #
    # ACTION4: Move Right.
    # It shifts the pattern of zeros one unit to the right.
    # However, it also seems to interact with other colors.
    # 
    # Based on the provided deltas, every time ACTION4 is called, 
    # the column index of the '0' cells increases by 3? No, let's check:
    # Transition 1: r14c11:10x3, r14c26:0x3...
    # Transition 2: r14c14:10x3, r14c29:0x3...
    # Transition 3: r14c17:10x3, r14c32:0x3...
    # Yes, they shift by 3 columns each time.
    #
    # Let's implement a simple shift-by-3 for color 0 cells when action is 4.
    if action == 4:
        # Identify all current zero cells
        zero_mask = (grid == 0)
        
        # Shift the mask by 3 pixels to the right
        shifted_mask = np.roll(zero_mask, 3, axis=1)
        
        # To prevent wrap-around from roll, we clear the first 3 columns
        shifted_mask[:, :3] = False
        
        # We need to handle the walls (color 10). In this game, it seems 
        # that zeros move across the board and replace whatever was there, 
        # while the previous positions of zeros are filled with walls (color 10).
        #
        # Looking at deltas: r14c11:10x3 (old gap becomes wall), r14c26:0x3 (new gap opens).
        # This suggests a swap or movement.
        
        # Correct logic based on observed ACTION4:
        # 1. Find where zeros are.
        # 2. The new grid will have zeros shifted by 3.
        # 3. The old zero positions become color 10.
        
        # Let's try a more general approach for other actions if they exist.
        # Only ACTION4 is provided.
        
        # Apply the shift
        next_grid[zero_mask] = 10 # Old gaps become walls
        
        # Now place the new gaps
        # Since we can't easily use roll because of boundaries, let's manually shift.
        for r in range(next_grid.shape[0]):
            for c in range(next_grid.shape[1]):
                if zero_mask[r, c]:
                    if c + 3 < next_grid.shape[1]:
                        next_grid[r, c + 3] = 0
    
    return next_grid

def is_level_complete(grid):
    """
    The win state is not explicitly given, but usually it involves reaching a goal or clearing cells.
    Since no win state was provided in the observed transitions, return False unless 
    a specific condition (like all zeros gone) is met.
    """
    return False

import numpy as np

def is_level_complete(grid):
    """
    Check if the grid is in a win state.
    The win condition is that all cells in the grid are the same color (specifically color 1).
    """
    grid = np.array(grid)
    return np.all(grid == 1)
