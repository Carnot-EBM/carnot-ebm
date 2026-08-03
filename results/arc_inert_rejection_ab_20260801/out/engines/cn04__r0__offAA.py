import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the same rules induced from observed transitions.
    ACTION 4 is a rightward shift of specific patterns in the same row/column range.
    """
    new_grid = grid.copy()
    if action == 4:
        # The pattern shifts right by 1 column.
        # We identify the region where changes occur (rows 14-31 and columns 11-63)
        # # Note: In the provided logs, ACTION 4 causes a sequence of cells to change values.
        # Based on the observations, it's a movement of 'empty' or 'wall' blocks.
        # Shift everything in rows 14-31 right by 1 pixel if possible.
        # For each row in that range, we find the pixels that are not background color 10.
        # 10 is the background.
        # 20 is often used for walls.
        # 10 is the background.
        # 10 is the uma own wall?
        # Let's look at the<|channel>thought  // No analysis allowed.
        
        # Simplified logic: Action 4 moves a set of objects right.
        # Find all non-background (color 10) cells in the active area.
        # Move them one step right.
        
        active_rows = slice(14, 32)
        active_cols = slice(11, 64)
        
        # We need to be careful about what shifts.
        # Looking at the deltas: r14c11 becomes 10, r14c26 becomes 0.
        # This looks like a shift of the "hole" (color 0).
        
        # Create a mask of the region being shifted.
        region = grid[active_rows, active_cols]
        shifted_region = np.full_like(region, 10) # Fill with background
        
        # Shift values from index i to i+1
        shifted_region[:, 1:] = region[:, :-1]
        
        # Apply back to new_grid
        new_grid[active_rows, active_cols] = shifted_region
        
        # Special case for top row observed in logs: r0c16:0x1 etc.
        # It seems some pixels on row 0 also change.
        # Let's check if they are just shifting too.
        # r0c16 -> 0, then r0c17 -> 0... this is a single pixel moving right.
        row0_val = grid[0, 16]
        # The log shows r0c16:0x1, then next action r0c17:0x1.
        # This means at t=0, cell [0,16] becomes 0. At t=1, cell [0,17] becomes 0.
        # Wait, the delta says "changed cells". If r0c16 was 4 and became 0, that's a change.
        # Then if r0c17 was 4 and became 0, and r0c16 returned to 4, that's two changes.
        # But the log only says r0c17:0x1. This implies r0c16 stayed 0? No, usually deltas are full.
        # Actually, looking closely: ACTION 4 (level 0->0) changed cells = r0c16:0x1 ...
        # Next ACTION 4 changed cells = r14c14:10x3 ... (no mention of row 0).
        # Third ACTION 4 changed cells = r0c17:0x1 ...
        # This suggests row 0 is updated every other turn or something.
        # Let's stick to the main region shift.
    
    return new_grid

def is_level_complete(grid):
    # Win state not provided in observations, but typically it involves reaching a goal.
    # Since we don't have one, return False.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for 'cn04' is that all cells of the same color 
    must be connected (either orthogonally or diagonally) and 
    there must be only one connected component of each color present in the grid.
    """
    grid = np.array(grid)
    unique_colors = np.unique(grid)
    
    # Remove background color (usually 0) if it's treated as empty space
    # However, for this specific task, we often treat all colors as part of the same rule.
    # Let's assume all colors present must be connected.
    
    for color in unique_colors:
        if color == 0: continue # Skip background
        
        # Find all cells of this color
        cells = np.argwhere(grid == color)
        if len(cells) == 0: continue
        
        # Use a simple BFS to find the connected component
        start_node = tuple(cells[0])
        visited = {start_node}
        queue = [start_node]
        
        while queue:
            curr = queue.pop(0)
            r, c = curr
            # Check 8-connectivity (orthogonal + diagonal)
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        if grid[nr, nc] == color and (nr, nc) not in visited:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
        
        # If the number of visited cells equals the total number of cells of that color,
        # it's a single connected component.
        if len(visited) != len(cells):
            return False
            
    return True
