import numpy as np

def engine(grid, action, data):
    res = grid.copy()
    
    # 1. Handle Marker at Column 63
    marker_row = -1
    for r in range(64):
        if grid[r, 63] == 5:
            marker_row = r
            break
    
    # If no marker found, we assume it starts above the first observed change (which was r3)
    # But for a general model, let's just find the current one.
    if marker_row != -1:
        res[marker_row, 63] = 11 # Reset old marker to background color of that col
    
    new_marker_row = (marker_row + 1) if marker_row != -1 else 0
    if new_marker_row < 64:
        res[new_marker_row, 63] = 5

    # 2. Identify Movable Blocks (not walls 10, 11 and not background 9)
    movables = []
    for r in range(64):
        for c in range(64):
            if grid[r, c] not in [9, 10, 11]:
                movables.append((r, c, grid[r, c]))
    
    # Clear movables from result before repositioning them
    for r, c, v in movables:
        res[r, c] = 9
        
    # 3. Move blocks based on action
    if action == 2: # Down
        for r, c, v in movables:
            nr, nc = r + 3, c
            if nr < 64:
                res[nr, nc] = v
    elif action == 3: # Left
        for r, c, v in movables:
            nr, nc = r, c - 3
            if nc >= 0:
                res[nr, nc] = v
    else:
        # For other actions, just put them back where they were
        for r, c, v in movables:
            res[r, c] = v

    # 4. Restore Walls (ensure walls are never overwritten)
    for r in range(64):
        for c in range(64):
            if grid[r, c] in [10, 11]:
                res[r, c] = grid[r, c]
                
    return res

def is_level_complete(grid):
    return False