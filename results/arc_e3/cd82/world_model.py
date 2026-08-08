import numpy as np

def engine(grid, action, data):
    """
    The world model for the ARC-AGI-3 game 'cd82'.
    The game involves moving a block (color 2 with color 15 inside) around a stationary 
    obstacle (color 0) to reach a target position (below the obstacle).
    """
    # Action mapping: 1: Up, 2: Down, 3: Left, 4: Right
    if action == 1:
        dy, dx = -1, 0
    elif action == 2:
        dy, dx = 1, 0
    elif action == 3:
        dy, dx = 0, -1
    elif action == 4:
        dy, dx = 0, 1
    else:
        return grid.copy()

    # Find the bounding box of the moving block (color 2)
    rows, cols = np.where(grid == 2)
    if len(rows) == 0:
        return grid.copy()
    
    y0, x0 = np.min(rows), np.min(cols)
    y1, x1 = np.max(rows), np.max(cols)
    
    # Find the bounding box of the stationary obstacle (color 0)
    rows0, cols0 = np.where(grid == 0)
    if len(rows0) == 0:
        # If no obstacle, the block can move freely
        y0_0, x0_0, y1_0, x1_0 = -1, -1, -1, -1
    else:
        y0_0, x0_0 = np.min(rows0), np.min(cols0)
        y1_0, x1_0 = np.max(rows0), np.max(cols0)

    # Calculate new bounding box for the moving block
    ny0, nx0 = y0 + dy, x0 + dx
    ny1, nx1 = y1 + dy, x1 + dx

    # Check grid boundaries
    if ny0 < 0 or ny1 >= grid.shape[0] or nx0 < 0 or nx1 >= grid.shape[1]:
        return grid.copy()

    # Check for overlap with the obstacle (color 0)
    # Overlap occurs if the new block bounding box intersects the obstacle bounding box
    if not (ny1 < y0_0 or ny0 > y1_0 or nx1 < x0_0 or nx0 > x1_0):
        return grid.copy()

    # Move the block:
    # 1. Clear the old block area (color 2 and color 15)
    # We use the bounding box of color 2 to clear everything inside it.
    new_grid = grid.copy()
    new_grid[y0:y1+1, x0:x1+1] = 5
    
    # 2. Draw the new block area
    # Outer block (color 2)
    new_grid[ny0:ny1+1, nx0:nx1+1] = 2
    
    # Inner block (color 15)
    # The inner block is offset by 1 from the outer block's boundaries
    # Initial: Block 2 (24, 25, 32, 38), Block 15 (25, 26, 31, 37)
    # Offset: (1, 1) and size is (y1-y0-1, x1-x0-1)
    new_grid[ny0+1:ny1, nx0+1:nx1] = 15

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the moving block (color 2) is positioned below 
    the stationary obstacle (color 0).
    """
    rows2, cols2 = np.where(grid == 2)
    rows0, cols0 = np.where(grid == 0)
    
    if len(rows2) == 0 or len(rows0) == 0:
        return False
    
    y0_2 = np.min(rows2)
    y1_0 = np.max(rows0)
    
    # Level is complete if the top of the block (color 2) is below the bottom of the obstacle (color 0)
    return y0_2 > y1_0