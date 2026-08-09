import numpy as np

import numpy as np

def engine(grid, action, data):
    """
    Predicts the next state of the grid given an action and its associated data.
    Action 6 is a click at (px, py) that changes a 6x6 block to color 8 and adds
    two pixels of color 11 to the bottom row (row 63), moving from right to left.
    """
    next_grid = grid.copy()
    if action == 6:
        px = data['x']
        py = data['y']
        
        # Change a 6x6 area starting at (py, px) to color 8
        # The observed transitions show blocks are exactly 6x6.
        for r in range(py, min(py + 6, 64)):
            for c in range(px, min(px + 6, 64)):
                next_grid[r, c] = 8
        
        # Update the bottom row (row 63) with two cells of color 11.
        # These cells fill from right to left: [62,63], then [60,61], etc.
        # We determine the current count by counting existing color 11s in row 63.
        current_11s = np.sum(grid[63, :] == 11)
        num_clicks = current_11s // 2
        start_col = 63 - 2 * (num_clicks + 1)
        end_col = 63 - 2 * num_clicks
        
        if start_col >= 0:
            # Set two consecutive cells to color 11
            # Based on observations: Click 1 -> col 62,63; Click 2 -> col 60,61...
            # The logic is: first click fills index 62 and 63.
            # Let's refine based on observed deltas:
            # Transition 1: r63c62:11x2  -> cols 62, 63
            # Transition 2: r63c60:11x2  -> cols 60, 61
            # Transition 3: r63c58:11x2  -> cols 58, 59
            next_grid[63, 62 - 2 * num_clicks] = 11
            next_grid[63, 63 - 2 * num_clicks] = 11

    return next_grid

def is_level_complete(grid):
    """
    Returns True if the level is complete.
    Based on the transitions, the level completes after the 4th successful block change.
    This corresponds to having 8 cells of color 11 in the bottom row (row 63).
    """
    # Count pixels of color 11 in the last row.
    count_11 = np.sum(grid[63, :] == 11)
    return count_11 >= 8

import numpy as np

def is_level_complete(grid):
    """
    Induces the win condition for ARC-AGI-3 game 'ft09'.
    The win condition for this 'fill' task is typically that all interior 
    areas enclosed by a boundary (non-zero cells) are completely filled 
    (contain no background/zero cells).
    """
    grid = np.array(grid)
    rows, cols = grid.shape
    if rows == 0 or cols == 0:
        return True
    
    # We define 'outside' as any cell that can reach the edge of the grid
    # without crossing a non-zero cell (the boundary).
    outside = np.zeros((rows, cols), dtype=bool)
    stack = []
    
    # Initialize the flood fill with all zero-cells on the edges
    for r in range(rows):
        for c in [0, cols - 1]:
            if grid[r, c] == 0:
                stack.append((r, c))
                outside[r, c] = True
    for c in range(cols):
        for r in [0, rows - 1]:
            if grid[r, c] == 0:
                stack.append((r, c))
                outside[r, c] = True
    
    # Flood fill to find all zero-cells connected to the edge
    while stack:
        r, c = stack.pop()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if not outside[nr, nc] and grid[nr, nc] == 0:
                    outside[nr, nc] = True
                    stack.append((nr, nc))
    
    # The level is complete if there are no zero-cells that are 'inside' 
    # (i.e., zero-cells that cannot reach the edge without crossing the boundary).
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] == 0 and not outside[r, c]:
                return False
                
    return True
