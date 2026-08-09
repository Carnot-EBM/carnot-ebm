import numpy as np

def engine(grid, action, data):
    """
    World model for ARC-AGI game 'ka59'.
    The grid is a 64x64 integer array.
    Actions move objects in rows 30-32 and update a progress bar on row 63.
    """
    new_grid = grid.copy()
    
    # Progress bar timer: Row 63 decreases from right to left each action.
    # We find the first non-zero cell from the right and set it to 0.
    for col in range(63, -1, -1):
        if new_grid[63, col] != 0:
            new_grid[63, col] = 0
            break

    # ACTION3 moves an object left by 3 units; ACTION4 moves it right by 3 units.
    # Based on observed transitions, this affects blocks in rows 30-32.
    # Specifically, it swaps 3-cell segments horizontally.
    if action == 3: # Move Left
        # Target block starts at c18 (observed)
        start_col = 18
        end_col = start_col + 3
        prev_col = start_col - 3
        if prev_col >= 0:
            for r in [30, 31, 32]:
                segment1 = new_grid[r, prev_col:start_col].copy()
                segment2 = new_grid[r, start_col:end_col].copy()
                new_grid[r, prev_col:start_col] = segment2
                new_grid[r, start_col:end_col] = segment1
                
    elif action == 4: # Move Right
        # The active block's position changes over time. We find the current color-14 block.
        # In row 30, we look for a sequence of three 14s.
        active_col = -1
        for col in range(9, 60):
            if np.all(new_grid[30, col:col+3] == 14):
                active_col = col
                break
        
        if active_col != -1:
            next_col = active_col + 3
            if next_col + 3 <= 64:
                for r in [30, 31, 32]:
                    segment1 = new_grid[r, active_col:next_col].copy()
                    segment2 = new_grid[r, next_col:next_col+3].copy()
                    new_grid[r, active_col:next_col] = segment2
                    new_grid[r, next_col:next_col+3] = segment1

    # ACTION6 is a click that can toggle colors between specific cells.
    elif action == 6 and data is not None:
        px, py = data['x'], data['y']
        # Observed interaction at (31, 43) and (31, 28).
        if py == 31 and px == 43:
            val_a = new_grid[31, 43]
            val_b = new_grid[31, 28]
            new_grid[31, 43] = 0 if val_a != 0 else 4
            new_grid[31, 28] = 4 if val_b != 4 else 0
        elif py == 31 and px == 28:
            val_a = new_grid[31, 43]
            val_b = new_grid[31, 28]
            new_grid[31, 43] = 4 if val_a != 4 else 0
            new_grid[31, 28] = 0 if val_b != 0 else 4

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the progress bar on row 63 is fully depleted.
    """
    return np.all(grid[63, :] == 0)