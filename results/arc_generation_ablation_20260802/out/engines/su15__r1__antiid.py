import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game appears to be a puzzle where clicking (ACTION6) 
    # on a cell changes its neighborhood into a color 15 (magenta) block
    # and reverts previous blocks back to their background colors.
    # It's also some kind//of progress indicator moving along row 63.
    
    if action == 6:
        x, y = data['x'], data['y']
        
        # Create a copy of the grid to maintain purity
        new_grid = grid.copy()
        
        # 3x3 block of color 15 centered at (y, x)
        # Note: logical coordinates are used; pixels are 1:1 here.
        # We apply the same pattern seen in the observed transitions.
        # The new block becomes 15
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                # Bounds check
                if 0 <= y + dy < 64 and 0 <= x + dx < 64:
                    new_grid[y + dy, x + dx] = 15
        
        # Based on the observations, it's thes specific cells that change.
        # The laest click is always replacing something else.
        # The same logic seems to be be reverting a previous block.
        # Find existing blocks of 15s and revert them to background.
        # For example, if wesre clicking near a 15-block, 
        #<|channel>thought
        #<|channel>thought
        # find all 3x3 areas that were 15s and turn them back to background.
        # find any cell that was 15 and not part of the current click area.
        # 
        # The background colors vary by region (color 4 or 5).
        # Background color for row 4-6 are 4, rows 10-62 are 5.
        # Row 63 is 0.
        # 
        # Let's implement a simple rule:
        # 1. Identify the target 3x3 block centered at (y, x)
        # 2. Revert all other 15s in the grid to their "natural" background.
        # 3. Set the target 3x3 block to 15.
        
        # First, identify natural backgrounds
        bg = np.full((64, 64), 5, dtype=int)
        bg[0:4, :] = 4
        bg[4:7, :] = 4 # based on r4:4x30,15x3,4x31 etc
        bg[7:10, :] = 4
        bg[63, :] = 0
        
        # Special case for the diagonal line of 3s observed in initial grid
        # We only revert 15s that aren't part of the current click.
        # To be safe and match the delta exactly, we restore background colors.
        # The observations show a specific pattern of reverting blocks.
        # 
        # Let's refine the background restoration:
        # Any cell that is currently 15 and not in the new 3x3 area should be restored.
        # What color? In the deltas, they are restored to 5 (mostly).
        # For row 58-60, it's also 5.
        
        for r in range(64):
            for c in range(64):
                if grid[r, c] == 15:
                    # If this cell is NOT in the new 3x3 block, revert it.
                    if not (abs(r - y) <= 1 and abs(c - x) <= 1):
                        # Restore based on basic region rules
                        if r < 10:
                            new_grid[r, c] = 4
                        elif r >= 63:
                            new_grid[r, c] = 0
                        else:
                            new_grid[r, c] = 5
        
        # Now set the current click target to 15
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if 0 <= y + dy < 64 and 0 <= x + dx < 64:
                    new_grid[y + dy, x + dx] = 15
        
        # Progress indicator on row 63:
        # The deltas show r63c62:5x2, then r63c60:5x2, etc.
        # It moves left by 2 cells each time.
        # Let's find where the progress bar is currently.
        # Row 63 starts as all 0s.
        # First action: r63c62:5x2 (cols 62, 63 become 5)
        # Second action: r63c60:5x2 (cols 60, 61 become 5)
        # Third action: r63c58:5x2 (cols 58, 59 become 5)
        # This means it fills from right to left.
        
        # Find current filled length of row 63
        filled_count = np.sum(grid[63, :] == 5)
        next_start_col = 63 - (filled_count + 2)
        if next_start_col >= 0:
            new_grid[63, next_start_col : next_start_col + 2] = 5
            
        return new_grid

    return grid

def is_level_complete(grid):
    # Level complete when the progress bar on row 63 is full?
    # Or some other condition. Given no win state provided, we guess.
    return np.all(grid[63, :] == 5)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state for the 'su15' game.
    The win state is typically a sorted sequence of numbers (1 to N-1) 
    followed by a blank space (0).
    """
    grid = np.array(grid)
    flat = grid.flatten()
    
    # The target sequence for a 15-puzzle style game: 1, 2, ..., 15, 0
    # Adjust the target based on the actual grid size
    n = flat.size
    target = np.arange(1, n)
    target = np.append(target, 0)
    
    return np.array_equal(flat, target)
