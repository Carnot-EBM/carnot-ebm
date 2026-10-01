import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track progress on the rightmost column (col 63)
    # Find current highest row that has color 5 in col 63
    col63 = g[:, 63]
    filled_rows = np.where(col63 == 5)[0]
    if len(filled_rows) > 0:
        next_row = int(np.max(filled_rows)) + 1
    else:
        next_row = 0
    
    # Actions 1-5 advance the counter by 1
    if action in [1, 2, 3, 4, 5]:
        if next_row < 64:
            g[next_row, 63] = 5
    
    # Action 6: click - no change observed
    if action == 6:
        pass
    
    return g

def is_level_complete(grid):
    # Check if all cells in column 63 are filled with color 5
    col63 = grid[:, 63]
    return np.all(col63 == 5)