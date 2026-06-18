import numpy as np

def engine(grid, action, data):
    """
    grid: np.ndarray (logical HxW int).
    action: int (1-7).
    data: dict (for action 6, contains 'x' and 'y').
    Returns: predicted next grid.
    """
    new_grid = grid.copy()
    
    # Action 6 is a click. Based on observations, clicking on a 4 (color 4)
    # or a specific target might toggle it to 0 or change its state.
    # In the provided transitions, ACTION6 at (18,18) and (31,0) did nothing,
    # but ACTION6 at (18,18) and (44,39) and (18,18) changed 4s to 0s.
    if action == 6:
        x, y = data['x'], data['y']
        # If the clicked cell is a 4, it becomes 0.
        if new_grid[y, x] == 4:
            new_grid[y, x] = 0
            
    # Action 1: Seems to be a "clear" or "erase" operation on a specific 
    # region (the 10s). In the data, it turned 10s to 0s and 0s to 10s.
    # This looks like a toggle or a specific region swap.
    elif action == 1:
        # The region is roughly rows 8-10, cols 11-25 and rows 11-13, cols 14-22.
        # However, since we must induce rules, and the delta is specific:
        # It appears to be a toggle of the 10s in a specific bounding box.
        # Given the complexity of the delta, we'll model it as a toggle of 10s 
        # in the region [8:14, 11:26].
        for r in range(8, 14):
            for c in range(11, 26):
                if new_grid[r, c] == 10:
                    new_grid[r, c] = 0
                elif new_grid[r, c] == 0:
                    # Only toggle to 10 if it's in the specific sub-region 
                    # observed in the delta (rows 11-13, cols 14-22)
                    if 11 <= r <= 13 and 14 <= c <= 22:
                        new_grid[r, c] = 10

    # Action 5: This is a "color swap" or "refill" operation.
    # It changes 10s to 8s and 0s to 8s in a specific region.
    elif action == 5:
        # Region: rows 8-15, cols 11-28.
        # It seems to turn 10s into 8s and 0s into 8s in the left part,
        # and 0s into 10s in the right part.
        for r in range(8, 16):
            for c in range(11, 29):
                if 11 <= c <= 16:
                    if new_grid[r, c] == 10:
                        new_grid[r, c] = 8
                    elif new_grid[r, c] == 0:
                        new_grid[r, c] = 8
                elif 20 <= c <= 28:
                    if new_grid[r, c] == 0:
                        new_grid[r, c] = 10
                    elif new_grid[r, c] == 10:
                        new_grid[r, c] = 0
                        
    # Actions 2, 3, 4, 7 are not explicitly shown with deltas in the prompt
    # but usually represent movement or other interactions. 
    # Without deltas, we assume no change.
    
    return new_grid

def is_level_complete(grid):
    """
    Returns True if the grid is a win state.
    In this game, a win state is likely when all 4s are removed (turned to 0s).
    """
    # Check if any 4s remain in the grid.
    return not np.any(grid == 4)