import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state given an action.
    Based on observed transitions for game 'lp85'.
    """
    # Action 0 advances the level by changing blocks of 5 cells in column 0 to color 5
    # and shifting the colors of specific 4x4 object blocks.
    if action == 0:
        new_grid = grid.copy()
        
        # Rule 1: Column 0 progression. Every ACTION0 changes the next 5 cells from 14 to 5.
        # Find the first cell in column 0 that is still color 14 (or not 5).
        for r in range(64):
            if new_grid[r, 0] != 5:
                # Change this cell and the next 4 to color 5.
                for i in range(r, min(r + 5, 64)):
                    new_grid[i, 0] = 5
                break
        
        # Rule 2: Object Color Shifting.
        # The objects are located at specific coordinates and shift their colors.
        # Based on observations, we identify these as a set of 4x4 blocks.
        objects = [
            (19, 12), (19, 18), (19, 24), (19, 30), (19, 36), (19, 42), (19, 48),
            (25, 12), (25, 48),
            (31, 12), (31, 48),
            (37, 12), (37, 48),
            (43, 12), (43, 18), (43, 24), (43, 30), (43, 36), (43, 42), (43, 48)
        ]
        
        # Extract current colors from the top-left cell of each object block.
        current_colors = [grid[r, c] for r, c in objects]
        
        # Shift colors to the left by one position.
        # The observed shift is slightly complex, but a simple rotation often fits these puzzles.
        shifted_colors = current_colors[1:] + [current_colors[0]]
        
        # Apply shifted colors back to the grid blocks.
        for i, (r, c) in enumerate(objects):
            color = shifted_colors[i]
            new_grid[r:r+4, c:c+4] = color
            
        return new_grid

    return grid

def is_level_complete(grid):
    """
    Determines if the level is complete based on the state of the grid.
    The win transition occurs after ACTION0 has been applied 5 times,
    meaning column 0 contains at least 25 cells of color 5.
    """
    # Count how many cells in column 0 are color 5.
    count_five = np.sum(grid[:, 0] == 5)
    
    # Based on observations, the winning move makes col 0 have 25 cells of color 5.
    return count_five >= 25