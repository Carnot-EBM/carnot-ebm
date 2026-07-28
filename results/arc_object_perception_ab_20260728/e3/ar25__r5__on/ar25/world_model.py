import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 3:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 15:
            return grid
        if py >= 15 and py < 24:
            if px < 6:
                return grid
            if px >= 6 and px < 15:
                return grid
            if px >= 15 and px < 45:
                return grid
            if px >= 45 and px < 54:
                return grid
            if px >= 54:
                return grid
        if py >= 24:
            return grid
    elif action == 2:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 18:
            return grid
        if py >= 18 and py < 24:
            if px < 3:
                return grid
            if px >= 3 and px < 9:
                return grid
            if px >= 9 and px < 12:
                return grid
            if px >= 12 and px < 21:
                return grid
            if px >= 21 and px < 24:
                return grid
            if px >= 24:
                return grid
        if py >= 24:
            return grid
    elif action == 1:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        if py < 18:
            return grid
        if py >= 18 and py < 24:
            if px < 3:
                return grid
            if px >= 3 and px < 9:
                return grid
            if px >= 9 and px < 12:
                return grid
            if px >= 12 and px < 21:
                return grid
            if px >= 21 and px < 24:
                return grid
            if px >= 24:
                return grid
        if py >= 24:
            return grid
    return grid

def is_level_complete(grid):
    H, W = grid.shape
    for r in range(H):
        row_str = ""
        for c in range(W):
            val = grid[r, c]
            if row_str == "":
                row_str += f"{val}x1"
            elif val == int(row_str.split('x')[0]):
                row_str += f"x1"
            else:
                row_str += f",{val}x1"
        if row_str != "r" + str(r) + ":" + row_str:
            return False
    return True

def is_level_complete(grid):
    import numpy as np
    grid = np.array(grid)
    if grid.shape != (10, 10):
        return False
    if grid.dtype != object:
        return False
    if not np.all([isinstance(cell, str) for row in grid for cell in row]):
        return False
    if not np.all([cell in ['.', 'o', 'X', 'O'] for row in grid for cell in row]):
        return False
    if not np.all([row[0] == row[1] == row[2] == row[3] == row[4] == row[5] == row[6] == row[7] == row[8] == row[9] for row in grid]):
        return False
    if not np.all([grid[i][0] == grid[i][1] == grid[i][2] == grid[i][3] == grid[i][4] == grid[i][5] == grid[i][6] == grid[i][7] == grid[i][8] == grid[i][9] for i in range(10)]):
        return False
    if not np.all([grid[i][j] == grid[i][j+1] == grid[i][j+2] == grid[i][j+3] == grid[i][j+4] == grid[i][j+5] == grid[i][j+6] == grid[i][j+7] == grid[i][j+8] == grid[i][j+9] for i in range(10) for j in range(10)]):
        return False
    if not np.all([grid[i][j] == grid[i+1][j] == grid[i+2][j] == grid[i+3][j] == grid[i+4][j] == grid[i+5][j] == grid[i+6][j] == grid[i+7][j] == grid[i+8][j] == grid[i+9][j] for i in range(10) for j in range(10)]):
        return False
    return True
