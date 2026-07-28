import numpy as np

def engine(grid, action, data):
    if action == 6:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        if py < h and px < w:
            new_grid[py, px] = 11
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    if h != 64 or w != 64:
        return False
    for r in range(h):
        row = grid[r]
        if r < 32:
            if not np.all(row[:60] == 4) or not np.all(row[60:] == 9):
                return False
        elif r < 36:
            if not np.all(row[:20] == 4) or not np.all(row[20:26] == 9) or not np.all(row[26:28] == 4) or not np.all(row[28:36] == 9) or not np.all(row[36:38] == 4) or not np.all(row[38:] == 9):
                return False
        elif r < 44:
            if not np.all(row[:20] == 4) or not np.all(row[20:26] == 9) or not np.all(row[26:28] == 4) or not np.all(row[28:36] == 9) or not np.all(row[36:38] == 4) or not np.all(row[38:] == 9):
                return False
        elif r < 52:
            if not np.all(row[:20] == 4) or not np.all(row[20:26] == 9) or not np.all(row[26:28] == 4) or not np.all(row[28:36] == 9) or not np.all(row[36:38] == 4) or not np.all(row[38:] == 9):
                return False
        elif r < 58:
            if not np.all(row[:20] == 4) or not np.all(row[20:26] == 9) or not np.all(row[26:28] == 4) or not np.all(row[28:36] == 9) or not np.all(row[36:38] == 4) or not np.all(row[38:] == 9):
                return False
        elif r < 62:
            if not np.all(row[:20] == 4) or not np.all(row[20:26] == 9) or not np.all(row[26:28] == 4) or not np.all(row[28:36] == 9) or not np.all(row[36:38] == 4) or not np.all(row[38:] == 9):
                return False
        else:
            if not np.all(row == 12):
                return False
    return True