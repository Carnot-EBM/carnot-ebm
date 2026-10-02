import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track progress on the right wall (column 63)
    # Find the lowest row that has color 5 in column 63
    col63 = g[:, 63]
    five_rows = np.where(col63 == 5)[0]
    if len(five_rows) > 0:
        current_progress = int(np.max(five_rows))
    else:
        current_progress = -1
    
    # Actions 1-5 increment progress by 1 (move a marker down the right wall)
    if action in [1, 2, 3, 4, 5]:
        new_row = current_progress + 1
        if 0 <= new_row < 64 and g[new_row, 63] != 5:
            g[new_row, 63] = 5
    
    # Action 6 is a click; observed no change, so do nothing
    
    return g

def is_level_complete(grid):
    # The level is complete when all 64 cells in column 63 are filled with color 5
    # (i.e., the progress marker has reached the bottom of the grid)
    col63 = grid[:, 63]
    return np.all(col63 == 5)