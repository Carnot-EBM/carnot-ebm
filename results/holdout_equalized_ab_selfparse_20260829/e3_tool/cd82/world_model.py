import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape
    
    # Find the 2-colored border object (the moving entity)
    # It's a rectangular border of 2s with 15s inside
    mask2 = (g == 2)
    rows = np.where(mask2.any(axis=1))[0]
    cols = np.where(mask2.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return g
    
    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]
    
    # Determine movement direction
    dr, dc = 0, 0
    if action == 1:  # up
        dr = -1
    elif action == 2:  # down
        dr = 1
    elif action == 3:  # left
        dc = -1
    elif action == 4:  # right
        dc = 1
    
    if dr == 0 and dc == 0:
        return g
    
    # Extract the object (2-border + 15-interior)
    obj = g[r0:r1+1, c0:c1+1].copy()
    
    # New position
    nr0, nr1 = r0 + dr, r1 + dr
    nc0, nc1 = c0 + dc, c1 + dc
    
    # Check bounds
    if nr0 < 0 or nr1 >= H or nc0 < 0 or nc1 >= W:
        return g
    
    # Check collision: new position must not overlap with non-5, non-2, non-15 cells
    # (i.e., can't move into walls or other objects)
    new_region = g[nr0:nr1+1, nc0:nc1+1]
    # The object can move into 5 (background), 2, or 15 cells
    # It cannot move into 0, 3, 4, or other colors
    blocked = (new_region != 5) & (new_region != 2) & (new_region != 15)
    if blocked.any():
        return g
    
    # Clear old position (set to 5)
    g[r0:r1+1, c0:c1+1] = 5
    
    # Place object at new position
    g[nr0:nr1+1, nc0:nc1+1] = obj
    
    return g

def is_level_complete(grid):
    # Check if the 15-block has reached the 0-block area (rows 34-43)
    # or some other win condition
    # For now, return False as we don't have a win state observed
    return False