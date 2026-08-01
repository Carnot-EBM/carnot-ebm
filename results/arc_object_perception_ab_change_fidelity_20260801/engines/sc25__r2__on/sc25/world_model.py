import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # ACTION3 is a left movement for the player character/object.
    # The "player" seems to be the composite object consisting of colors 9, 10, and 2.
    # Based on observed transitions, when ACTION3 is triggered, the entire group moves left by 2 columns.
    # We identify the player' same-shape objects that move.
    # In the initial state, the player consists of obj3, obj4, obj5, obj6, obj7.
    # These are colors 9, 10, and 2.
    #<|channel>thought
    # Let's refine the player identification.
    # Looking at the delta: r19c37:9x1,10x1,2x2 -> r19c35:9x1,10x1,2x2.
    # This is exactly moving a pattern [9, 10, 2, 2] two units left.
    # Specifically, it looks like the block of color 2 (obj5) is shifting.
    # Also, some cells in column 62 (color 14) are changing to 0.
    # It appears there is a mechanism where moving the player triggers changes elsewhere.
    # The "player" object is the one containing colors 9, 10, 2.
    # The target area for the win condition might be related to these movements.

    if action == 3:
        # Find all pixels of colors 9, 10, 2 that form the moving entity.
        # In the observed data, only specific rows (19, 20) move.
        # We need to find the contiguous blocks of colors 9, 10, 2.
        new_grid = grid.copy()
        
        # Identify the moving part: specifically the block at r19-r20 and c37+
        # Looking at the deltas: ACTION3 moves the sequence [9, 10, 2, 2] from col 37 to 35, then 33, etc.
        # This is a shift of 2 units left per ACTION3 call.
        
        # Let's implement a general rule: any horizontal sequence of [9, 10, 2, 2] in rows 19-20 shifts left by 2.
        for r in [19, 20]:
            # Search for the pattern [9, 10, 2, 2]
            for c in range(63, 3, -1):
                if np.array_equal(grid[r, c:c+4], [9, 10, 2, 2]):
                    # Move it 2 cells left
                    new_grid[r, c-2:c+2] = grid[r, c:c+4] # this is wrong logic
                    # Correct move:
                    # new_grid[r, c-2:c+2] = [9, 10, 2, 2]
                    # The original position should be restored to background (color 5)
                    # But wait, if we move it left, the pixels at c and c+1 are now empty?
                    # No, looking at deltas: "r19c37:9x1,10x1,2x2" means col 37=9, 38=10, 39=2, 40=2.
                    # Then "r19c35:9x1,10x1,2x2" means col 35=9, 36=10, 37=2, 38=2.
                    # So the block moves from {37,38,39,40} to {35,36,37,38}.
                    # This is a shift of -2.
                    new_grid[r, c-2:c+2] = [9, 10, 2, 2]
                    new_grid[r, c+2:c+4] = 5 # restore background
        
        # Also handle the side effect: color 14 cells in column 62 turning into 0.
        # The observed transitions show r6c62, r7c62 then r8c62, r9c62...
        # It seems every ACTION3 call clears two rows of the wall at col 62.
        # We need to track how many ACTION3s have occurred or use the current state.
        # Let's find the first row where grid[r, 62] == 14 and clear it and the next one.
        for r in range(64):
            if grid[r, 62] == 14:
                new_grid[r, 62] = 0
                if r + 1 < 64 and grid[r+1, 62] == 14:
                    new_grid[r+1, 62] = 0
                break

        return new_grid

    return grid

def is_level_complete(grid):
    # Win condition: likely when all color 14 pixels are gone or player reaches a goal.
    # In the provided data, we don't see a win state, but usually it's clearing an object.
    return np.all(grid[:, 62] != 14)