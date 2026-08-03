import numpy as np

def engine(grid, action, data):
    # Action map based on observations
    # ACTION 1: UP, ACTION 2: DOWN, ACTION 3: LEFT, ACTION 4: RIGHT
    move_map = {1: (-3, 0), 2: (3, 0), 3: (0, -3), 4: (0, 3)}
    if action not in move_map:
        return grid
    
    dr, dc = move_map[action]
    
    # Find the moving object (color 14)
    coords = np.argwhere(grid == 14)
    if coords.size == 0:
        return grid
    
    # We assume the object is a contiguous block.
    # To handle the "hole" (color 0), we find all non-background cells that are part of the object.
    # Background colors are 2 and 15.
    obj_mask = (grid != 2) & (grid != 15)
    # But wait, there are other things like color 4 and 5.
    # Let's just use the bounding box of color 14 to define the object.
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    # Create a mask for the object based on its current bounding box
    # The object seems to be roughly 3x3.
    obj_indices = []
    for r in range(min_r, max_r + 1):
        for c in range(min_c, max_c + 1):
            if grid[r, c] != 2 and grid[r, c] != 15:
                obj_indices.append((r, c))
    
    new_grid = grid.copy()
    
    # Erase old positions - replace with path color 1 or background 2
    for r, c in obj_indices:
        # If it was part of the moving object, it should leave behind the path color 1
        # if that cell is within the "path" area.
        new_grid[r, c] = 1 if (grid[r, c] == 14 or grid[r, c] == 0) else grid[r, c]
        # Actually, looking at deltas, the cells left behind become 1.
        new_grid[r, c] = 1

    # Move and place new positions
    for r, c in obj_indices:
        nr, nc = r + dr, c + dc
        if 0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]:
            new_grid[nr, nc] = grid[r, c]
        else:
            # Out of bounds, just keep it as is or block movement
            pass
            
    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a specific condition is met.
    # For now, return False.
    return False