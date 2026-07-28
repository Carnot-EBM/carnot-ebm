import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    if action == 2:
        # Action 2: Toggle a specific region
        # Based on observed transitions, this action toggles a rectangular region
        # The region is defined by the changed cells pattern
        # We apply the delta directly
        # The delta is given in the observed transitions, but we need to infer the rule
        # From the data, it seems to toggle a region in the middle of the grid
        # Let's apply the delta as observed
        # Since we don't have the exact delta for this action, we need to infer it
        # Based on the pattern, it seems to toggle a region
        # We'll apply a simple toggle rule
        # The region seems to be around rows 11-31, columns 11-23
        # We'll toggle this region
        for r in range(11, 32):
            for c in range(11, 24):
                if grid[r, c] == 13:
                    new_grid[r, c] = 0
                else:
                    new_grid[r, c] = 13
    elif action == 5:
        # Action 5: Toggle another region
        # Based on observed transitions, this action toggles a different region
        # The region is around rows 11-31, columns 11-26
        for r in range(11, 32):
            for c in range(11, 27):
                if grid[r, c] == 13:
                    new_grid[r, c] = 0
                else:
                    new_grid[r, c] = 13
    elif action == 6:
        # Action 6: Click action with data
        # Based on observed transitions, this action toggles a region around the click point
        # The region is defined by the click coordinates
        px, py = data['x'], data['y']
        # The region seems to be a rectangle around the click point
        # We'll toggle a region around the click point
        # Based on the pattern, it seems to toggle a region of size 10x10 or similar
        # We'll toggle a region around the click point
        for r in range(max(0, py - 5), min(H, py + 6)):
            for c in range(max(0, px - 5), min(W, px + 6)):
                if grid[r, c] == 13:
                    new_grid[r, c] = 0
                else:
                    new_grid[r, c] = 13
    elif action == 3:
        # Action 3: Toggle another region
        # Based on observed transitions, this action toggles a different region
        # The region is around rows 29-49, columns 35-47
        for r in range(29, 50):
            for c in range(35, 48):
                if grid[r, c] == 13:
                    new_grid[r, c] = 0
                else:
                    new_grid[r, c] = 13
    elif action == 1:
        # Action 1: No change
        pass
    
    return new_grid

def is_level_complete(grid):
    # Based on observed transitions, the level is complete when all cells are 13
    # or when the grid matches a specific pattern
    # From the initial grid, it seems the goal is to have all cells be 13
    # We'll check if all cells are 13
    return np.all(grid == 13)