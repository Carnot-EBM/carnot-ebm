import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        new_grid[py, px] = 7
        new_grid[py, px-1] = 4
        new_grid[py, px+1] = 4
        new_grid[py-1, px] = 4
        new_grid[py+1, px] = 4
        new_grid[py-1, px-1] = 4
        new_grid[py-1, px+1] = 4
        new_grid[py+1, px-1] = 4
        new_grid[py+1, px+1] = 4
        return new_grid
    return grid

def is_level_complete(grid):
    h, w = grid.shape
    for r in range(h):
        row = grid[r]
        if r == 0:
            if not np.all(row == 7):
                return False
        elif r == 1:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 2:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 3:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 4:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 5:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 6:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 7:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 8:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 9:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 10:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 11:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 12:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 13:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 14:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 15:
            if not np.all(row == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 16:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 17:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 18:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 19:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 20:
            if not np.all(row[:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 21:
            if not np.all(row[:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 22:
            if not np.all(row[:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 23:
            if not np.all(row[:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 24:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 25:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 26:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 27:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 28:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 29:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 30:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 31:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 32:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 33:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 34:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 35:
            if not np.all(row[:12] == 0) or not np.all(row[12:] == 3):
                return False
        elif r == 36:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 37:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 38:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 39:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 40:
            if not np.all(row[:28] == 5) or not np.all(row[28:42] == 14) or not np.all(row[42:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 41:
            if not np.all(row[:28] == 5) or not np.all(row[28:42] == 14) or not np.all(row[42:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 42:
            if not np.all(row[:28] == 5) or not np.all(row[28:42] == 14) or not np.all(row[42:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 43:
            if not np.all(row[:28] == 5) or not np.all(row[28:42] == 14) or not np.all(row[42:56] == 5) or not np.all(row[56:] == 3):
                return False
        elif r == 44:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 45:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 46:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 47:
            if not np.all(row[:4] == 9) or not np.all(row[4:52] == 0) or not np.all(row[52:] == 3):
                return False
        elif r == 48:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 49:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 50:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 51:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 52:
            if not np.all(row[:8] == 0) or not np.all(row[8:22] == 14) or not np.all(row[22:26] == 4) or not np.all(row[26:] == 3):
                return False
        elif r == 53:
            if not np.all(row[:8] == 0) or not np.all(row[8:22] == 14) or not np.all(row[22:26] == 4) or not np.all(row[26:] == 3):
                return False
        elif r == 54:
            if not np.all(row[:8] == 0) or not np.all(row[8:22] == 14) or not np.all(row[22:26] == 4) or not np.all(row[26:] == 3):
                return False
        elif r == 55:
            if not np.all(row[:8] == 0) or not np.all(row[8:22] == 14) or not np.all(row[22:26] == 4) or not np.all(row[26:] == 3):
                return False
        elif r == 56:
            if not np.all(row[:8] == 0) or not np.all(row[8:22] == 14) or not np.all(row[22:26] == 4) or not np.all(row[26:] == 3):
                return False
        elif r == 57:
            if not np.all(row[:8] == 0) or not np.all(row[8:22] == 14) or not np.all(row[22:26] == 4) or not np.all(row[26:] == 3):
                return False
        elif r == 58:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 59:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 60:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 61:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 62:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
        elif r == 63:
            if not np.all(row[:8] == 0) or not np.all(row[8:] == 3):
                return False
    return True