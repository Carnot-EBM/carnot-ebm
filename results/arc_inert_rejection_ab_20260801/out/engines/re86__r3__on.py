import numpy as np

import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION4: Right movement of a specific object/column pattern.
    # ACTION5: Left movement or state change.
    # ACTION1: Upward movement of a specific object/column pattern.
    # ACTION2: Downward movement? (Not inferred from data but symmetric to ACTION1).
    # ACTION3: Left movement? (Symmetric to ACTION4).
    # ACTION 6 is click.
    
    new_grid = grid.copy()
    
    if action == 4: # Move Right
        # Identify the vertical bar at col 39-42 range and shift it right by 3 units
        # This seems to be a column of color 9s.
        mask = (grid == 9)
        # We only move the parts that are not blocked by other colors or boundaries.
        # Shift mask right by 3 pixels.
        shifted_mask = np.roll(mask, 3, axis=1)
        
        # The logic in the delta shows a very specific set of cells changing.
        # It's essentially moving a 'cursor' or 'player' block.
        # Find all indices where value is 9.
        coords = np.argwhere(mask)
        for r, c in coords:
            new_grid[r, c] = 5 # Reset old position to background
            new_grid[r, c + 3 if c + 3 < 64 else 0] = 9
            
    elif action == 5: # Move Left
        mask = (grid == 9)
        coords = np.argwhere(mask)
        for r, c in coords:
            new_grid[r, c] = 5
            new_grid[r, c - 3 if c - 3 < 0 else c - 3] = 9
            
    elif action == 1: # Move Up
        # Based on ACTION1 deltas, it's shifting patterns upward.
        # Shift mask of color 9 and other objects.
        # shift_amount = 3
        # This looks like a more complex movement pattern involving multiple blocks.
        # The samethought process for theC-shaped object.
        #<|channel>thought// No need to actually simulate full physics; just shift coordinates.
        # Find all indices where value is 9.
        # Mask for "player" or "active" block.
        # Let's try a simple vertical shift for everything that isn't background (5).
        # For simplicity, we use the observed delta shifts.
        shift_val = 3
        mask = (grid != 5)
        # We only move things that are not fixed walls.
        # shifted_mask = np.roll(mask, -shift_val, axis=0)
        # {This is part of a specific game logic}
        
        # To be precise with the provided data, let's implement the shift based on the observed deltas.
        # In ACTION1, r13->r10, r24->r21, etc. which is a shift of -3 rows.
        coords = np.argwhere(grid != 5)
        for r, c in coords:
            new_grid[r, c] = 5
            if r - shift_val >= 0:
                new_grid[r - shift_val, c] = grid[r, c]
    
    elif action == 2: # Move Down
        shift_val = 3
        coords = np.argwhere(grid != 5)
        for r, c in coords:
            new_grid[r, c] = 5
            new_grid[r + shift_val if r + shift_val < 64 else r, c] = grid[r, c]

    return new_grid

def is_level_complete(grid):
    # Based on the INITIAL GRID and transitions, there is no explicit win state given.
    # However, usually it involves reaching a target or collecting items.
    # The bottom row (r63) has color 1s that are being replaced by background/other colors.
    # This suggests a "progress bar" or "collection" mechanism.
    # Check if all cells of a certain type are gone or a specific cell is reached.
    # return False as we don't have a WIN STATE grid to compare.
    # In many ARC games, completion is when a specific pattern is formed.
    # Let' same assume the progress bar at r63 is full.
    return np.all(grid[63, :] == 1) # Placeholder logic based on observed delta changes at r63.

def is_level_complete(grid):
    thought = "The same color pixels own the same region own the same color"
    # Check if all pixels of the same color are connected
    # This is a<|channel>thought process, not//
    # Simple rule: All pixels of same color must be connected.
    # Use BFS to find connected components of each color.
    import numpy as np
    grid = np.array(grid)
    rows, cols = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    
    colors = np.unique(grid)
    for color in colors:
        # Find first pixel of this color
        start_node = np.argwhere(grid == color)[0]
        
        # BFS to find all connected pixels of this color
        queue = [start_node]
        component = set()
        while queue:
            curr = queue.pop(0)
            curr_tuple = tuple(curr)
            if curr_tuple not in component:
                component.add(curr_tuple)
                # Check 4-neighbors
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = curr[0] + dr, curr[1] + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr, nc] == color and (nr, nc) not in component:
                        queue.append([nr, nc])
        
        # If the number of connected pixels is less than total pixels of this color, it's not complete
        if len(component) != np.sum(grid == color):
            return False
            
    return True
