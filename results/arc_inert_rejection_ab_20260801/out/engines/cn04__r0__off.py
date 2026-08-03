import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next grid state based on the same rules induced from observed transitions.
    ACTION 4 is a rightward shift of a specific pattern of cells (a 'cursor' or 'block').
    The cursor consists of a few regions that move together.
    Looking at the delta, ACTION 4 shifts everything in the region [14:32, 11:64] by +1 column.
    Wait, looking closer at the la-//’s deltas:
    r14c11:10x3 -> r14c14:10x3 -> r14c17:10x3 ...
    This is a movement of a 3-wide block of color 10.
    The blocks are moving synchronously across multiple rows and columns.
    Essentially, it looks like a "window" or "mask" of color 10 is shifting right.
    Actually, let's look at the top row: r0c16:0x1, then r0c17:0x1... this is a single pixel changing to 0.
    Let's analyze the coordinates of the shifted blocks.
    In each ACTION 4 transition, the x-coordinate of the changed areas increases by 3.
    """
    new_grid = grid.copy()
    if action == 4:
        # The observed transitions show that for every ACTION 4, 
        # a set of cells (originally starting around col 11) shifts right by 3 units.
        # Specifically, the pattern in rows 14-31 moves right by 3.
        # We need to identify which cells are part of the 'moving object'.
        # In the initial grid, there are regions of color 0, 8, 14.
        # Color 10 is the background/wall.
        # Let's define the region that moves. 
        # Based on the deltas, the movement happens in columns 11 to 63.
        # Row range is roughly 14 to 31.
        # For each row in that range, we shift the values from column 11 onwards.
        # To simulate this, we find the current position of the "gap" or "object".
        # Looking at the delta r14c11:10x3, r14c26:0x3... it means 
        # the block of 10s shifted into c11 and the 0s shifted out of c11.
        # The cursor is shifting right by 3 pixels.
        
        # Simplified rule: ACTION 4 shifts a specific set of blocks (the gaps) right by 3.
        # We identify the gap as any cell not equal to 10.
        # Find all cells != 10 in the rows 14-31.
        # Shift them right by 3.
        
        # Mask for the moving area
        mask_rows = slice(14, 32) # Approximate based on deltas
        mask_cols = slice(11, 64)
        
        # Create a temporary copy of the region
        region = grid[mask_rows, mask_cols].copy()
        
        # Shift right by 3
        shifted_region = np.full(region.shape, 10)
        shifted_region[:, 3:] = region[:, :-3]
        
        # Put it back
        new_grid[mask_rows, mask_cols] = shifted_region
        
        # Also handle the top row r0c16:0x1... which seems to be a progress bar or indicator.
        # Find the first color 0 pixel in row 0 from left to right.
        # Find its index.
        # Move that boundary one step to the right.
        row0 = new_grid[0, :].copy()
        first_zero = np.where(row0 == 0)[0]
        if len(first_zero) > 0:
            idx = first_zero[0]
            if idx < 63:
                new_grid[0, idx] = 0 # (Wait, the delta says r0c16:0x1, then r0c17:0x1...)
                # This means the cell at c16 becomes 0, then c17 becomes 0.
                # In INITIAL grid, r0 is 10x16, 4x32, 10x16. So cells [16, 47] are color 4.
                # The delta r0c16:0x1 means grid[0, 16] becomes 0.
                # Then r0c17:0x1 means grid[0, 17] becomes 0.
                # It's a "filling" process.
    
        # Correcting row 0 logic based on deltas:
        # ACTION 4: r0c16:0x1 -> r0c17:0x1 ...
        # Let's find the current 'filled' boundary in row 0.
        # Find the last index that was changed to 0.
        # Find the first index that is still color 4.
        row0_vals = grid[0, :]
        boundary = np.where(row0_vals == 4)[0]
        if len(boundary) > 0:
            idx = boundary[0]
            if idx < 63:
                new_grid[0, idx] = 0
        
        return new_grid

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the progress bar (row 0) is filled or the object reaches a goal.
    Based on the observed data, we only see ACTION 4 transitions.
    We don't have a win state grid.
    But usually, if the cursor moves far enough, it's complete.
    Looking at the progress bar in row 0, maybe it's complete when all cells [16, 47] are 0?
    """
    # Check if row 0 cells [16, 47] are all 0.
    return np.all(grid[0, 16:48] == 0)

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for 'cn04' is that all cells of the same color 
    (excluding background 0) are contiguous.
    """
    grid = np.array(grid)
    unique_colors = np.unique(grid[grid != 0])
    
    if len(unique_colors) == 0:
        return False
        
    for color in unique_colors:
        # Find all coordinates of the current color
        coords = np.argwhere(grid == color)
        if len(coords) == 0:
            return False
            
        # Start a BFS/DFS to find all reachable cells of the same color
        start_node = tuple(coords[0])
        visited = {start_node}
        stack = [start_node]
        
        while stack:
            curr = stack.pop()
            r, c = tuple(curr)
            # Check 4-connectivity
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = (r + dr, c + dc)
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    if grid[nr, nc] == color and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        stack.append((nr, nc))
        
        # If the number of visited cells equals the total number of cells of that color,
        # the color is contiguous.
        if len(visited) != len(coords):
            return False
            
    return True
