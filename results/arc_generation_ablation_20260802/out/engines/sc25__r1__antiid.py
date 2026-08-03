import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game state consists of a 64x64 grid.
    # Based on the observations, ACTION3 (likely 'left' movement) moves a set of objects.
    # There are two types of objects being moved:
    # 1. A moving object consisting of a small cluster of colors [9, 10, 2] at rows 19 and 20.
    # 2. A vertical bar of color 14 at column 62-63.
    # 3. Color 0 cells appearing at columns 62-63 replacing the 14s.
    #
    # Let's identify the "player" or "active object" in the same region.
    # In the initial grid, the player object is located around r19c35 to r20c38.
    # Looking at the transitions, ACTION3 shifts this object leftward by 2 units per call.
    #
    # Additionally, there_is a a side effect: every time ACTION3 is called, some blocks of color 14 at the right edge (col 62-63) are replaced by color 0.
    #
    # Rule for ACTION3:
    # - Shift the block of pixels [9, 10, 2, 2] at rows 19 and 20 left by 2 columns.
    # - Replace two rows of color 14 at col 62-63 with color 0.
    # - The rows being cleared are processed sequentially from top to bottom starting from row 6.
    # - Row indices for clearing: 6,7 then 8,9 then 10,11 etc.
    #
    # We need to keep track of state that might be part of the same level.
    # Since we don't have access to persistent state across calls to engine(), 
    # # we must infer it from the grid itself.
    # To find which rows of color 14 should be cleared, count how many times ACTION3 has been called.
    # To determine current position of the moving object, find its center.

    if action == 3:
        new_grid = grid.copy()
        
        # 1. Move the "player" object (colors 9, 10, 2)
        # Find all cells of colors 9, 10, or 2 in rows 19 and 20.
        mask = np.isin(grid[19:21, :], [9, 10, 2])
        coords = np.argwhere(mask)
        if coords.size > 0:
            # Shift left by 2 columns
            for r, c in coords:
                # Clear old position
                new_grid[r + 19, c] = 5 # Background color
                # Set new position
                nc = c - 2
                if nc >= 0:
                    new_grid[r + 19, nc] = grid[r + 19, c]
                else:
                    # Handle boundary
                    pass
            # We must apply shifts in order to avoid overwriting if wes shift into a cell
            # # that was<|channel>thought to be shifted.
            # # This is not a a bit-of-a-// la own logic.
            # Let's refine the movement.
            
        # Re-do movement more carefully
        player_rows = [19, 20]
        for row in player_rows:
            # Find indices of colors 9, 10, 2
            indices = np.where(np.isin(grid[row, :], [9, 10, 2]))[0]
            if indices.size > 0:
                # The object is at these indices.
                # Shift them left by 2 columns.
                # Clear old values first.
                for idx in indices:
                    new_grid[row, idx] = 5
                # Place new values
                for idx in indices:
                    val = grid[row, idx]
                    nc = idx - 2
                    if nc >= 0:
                        new_grid[row, nc] = val
                # Note: this might overwrite background color 5.
                # a a bit-of-a-// la own logic.

        # 2. Clear rows of color 14 at col 62-63
        # Count how many ACTION3 calls have happened (by counting zeros at the right edge)
        zeros_count = 0
        for r in range(64):
            if new_grid[r, 62] == 0 and new_grid[r, 63] == 0:
                zeros_count += 1
        
        # Next pair of rows to clear starting from row 6
        start_row = 6 + (zeros_count // 2) * 2
        if start_row < 64:
            new_grid[start_row, 62:64] = 0
            new_grid[start_row+1, 62:64] = 0 if start_row+1 < 64 else [0,0]
            
        return new_grid
    
    return grid

def is_level_complete(grid):
    # Level complete when all blocks of color 14 are gone or player reaches a goal.
    # Based on observations, we don't see a win state.
    # Let's assume it's complete when the right edge is cleared.
    return np.all(grid[:, 62:64] != 14)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the current grid state is a win state for the 'sc25' game.
    The win condition is defined as the grid containing only one color (excluding 0).
    """
    grid = np.array(grid)
    # Get all unique colors present in the grid, excluding the background color (0).
    unique_colors = np.unique(grid[grid != 0])
    
    # Win if there is exactly one unique color remaining on the grid.
    return len(unique_colors) == 1
