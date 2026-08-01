import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape)
    # Action 3 is a movement action that shifts an object composed of colors 9, 10, and 2
    # based on the observed transitions.
    # In the same frames, we look at thes cells changed in r19c37:9x1,10x1,2x2 etc.
    # The object consists of a sequence: color 9, then color 10, then color 2 twice.
    # 2x2 pixels of color 2.
    # The object being moved is the specific cluster of colors {9, 10, 2}.
    # Looking at the observations:
    # ACTION 3 moves the object leftward by 2 columns each time it's called.
    # # Let's find the object starting with color 9, followed by 10, then 2, 2.
    # # The object is located around rows 19 and 20.
    # #<|channel>thought
    # # In the initial grid, row 19 has "5x12, 9x1, 10x4, 2x22, 9x1, 10x1, 2x2, 5x19, 14x2"
    # # The target object is the one at col 37 (approx).
    # # The same happens for row 20.
    # # la////no_think
    # # The observed transitions show that Action 3 shifts this specific block [9, 10, 2, 2] in rows 19-20 leftwards.
    # # Additionally, some cells in column 62 are changing from 14 to 0.
    # # This looks like a progress bar or a timer linked to the movement.
    # # ACTION 3: move object left by 2 columns, change two cells of color 14 to 0 starting from top.
    
    new_grid = grid.copy()
    
    # Find the object [9, 10, 2, 2] in rows 19 and 20
    for r in [19, 20]:
        row = new_grid[r]
        # Search for sequence 9, 10, 2, 2
        for c in range(len(row) - 3):
            if row[c] == 9 and row[c+1] == 10 and row[c+2] == 2 and row[c+3] == 2:
                # Move it left by 2
                # We need to restore the background (color 5) at the old position
                # The block is 4 pixels wide.
                # Old positions: c, c+1, c+2, c+3
                # New positions: c-2, c-1, c, c+1
                if c >= 2:
                    # Restore original colors at current pos if they are not part of the move
                    # This is tricky because we might overwrite something.
                    # But based on observations, it's moving into color 5 area.
                    new_grid[r, c+2] = 5
                    new_grid[r, c+3] = 5
                    # Set new values
                    new_grid[r, c-2] = 9
                    new_grid[r, c-1] = 10
                    new_grid[r, c] = 2
                    new_grid[r, c+1] = 2
                break

    # Handle column 62 timer/progress bar
    # Find first occurrence of 14 in col 62 and change two cells to 0
    for r in range(64):
        if new_grid[r, 62] == 14:
            new_grid[r, 62] = 0
            # Try to find next 14 to turn to 0
            for r2 in range(r + 1, 64):
                if new_grid[r2, 62] == 14:
                    new_grid[r2, 62] = 0
                    break
            break
            
    return new_grid

def is_level_complete(grid):
    # Level complete if the object reaches a certain position or progress bar is gone.
    # In this specific case, we don't have the win state grid, but usually it's when 
    # the target object reaches its destination.
    return False # Default as no win state provided.

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition is based on the presence of a specific pattern or 
    the absence of certain elements.
    """
    grid = np.array(grid)
    # The win condition for sc25 is that all non-zero elements 
    # must be the same color.
    non_zero = grid[grid != 0]
    if non_zero.size == 0:
        return False
    return np.all(non_zero == non_zero[0])
