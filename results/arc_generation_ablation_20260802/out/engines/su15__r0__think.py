import numpy as np

def engine(grid, action, data):
    """
    The game appears to be a puzzle where clicking on specific locations (ACTION6)
    toggles the state of blocks or moves them. Based on the observed transitions:
    - Clicking at (x, y) changes a 3x3 area around that point to color 15.
    - Simultaneously, it seems to reset/change another 3x3 area back to color 5.
    - There's also a progress indicator updating in row 63.
    
    Looking closely at the deltas:
    Action 1: x=10, y=53 -> r52c9:15x3, r53c9:15x3, r54c9:15x3 AND r58c3:5x3...
    Action 2: x=16, y=47 -> r46c15:15x3, r47c15:15x3, r48c15:15x3 AND r52c9:5x3...
    This indicates a sequence where clicking one target activates it and potentially deactivates the previous one.
    """
    if action != 6:
        return grid.copy()

    new_grid = grid.copy()
    px, py = data['x'], data['y']
    
    # The click coordinates (px, py) correspond to the center of a 3x3 block.
    # In logical coords: row = py, col = px.
    # Based on delta: ACTION6 x=10, y=53 -> r52c9:15x3, r53c9:15x3, r54c9:15x3
    # This means the top-left is (py-1, px-1).
    
    # Activate current clicked area
    for r in range(py - 1, py + 2):
        for c in range(px - 1, px + 2):
            if 0 <= r < new_grid.shape[0] and 0 <= c < new_grid.shape[1]:
                new_grid[r, c] = 15
                
    # Deactivate previous areas or specific patterns observed in deltas
    # Observation shows that when Action N happens, the blocks from Action N-1 are reverted to color 5.
    # Since we don't have state memory across calls in this pure function, 
    # we look for existing 3x3 blocks of color 15 and revert them if they aren't the current one.
    
    # Find all 3x3 blocks of color 15
    for r in range(new_grid.shape[0] - 2):
        for c in range(new_grid.shape[1] - 2):
            if np.all(new_grid[r:r+3, c:c+3] == 15):
                # If this block is not the one we just created, reset it to 5
                if not (r == py - 1 and c == px - 1):
                    new_grid[r:r+3, c:c+3] = 5

    # Update row 63 progress indicator based on the pattern seen in deltas
    # r63c62:5x2 -> r63c60:5x2 -> r63c58:5x2 ...
    # It seems to be filling from right to left or moving a marker.
    # The delta shows "r63c62:5x2", then "r63c60:5x2". This suggests cells are being set to 5.
    # We'll simulate the specific observed change for these coordinates.
    # Since the exact logic of row 63 is obscure, we apply the most likely simple rule:
    # update the cell at (63, px*2) or similar. However, given the constraints, 
    # simply updating the grid as per the click is the primary mechanic.
    
    return new_grid

def is_level_complete(grid):
    """
    The win state isn't explicitly provided, but typically it involves 
    reaching a certain configuration or clearing all targets.
    Based on the patterns, we check if a specific condition is met.
    Since no WIN STATE was provided, we return False unless a known goal is reached.
    """
    # In ARC games, completion often means a specific pattern is formed.
    # Without a target grid, we assume the level is not complete yet.
    return False