import numpy as np

def engine(grid, action, data):
    if action == 1:
        # Push all objects of color 3, 5, 9, 11, 12 down by 1 row
        new_grid = grid.copy()
        for r in range(grid.shape[0] - 1):
            for c in range(grid.shape[1]):
                if grid[r, c] in [3, 5, 9, 11, 12]:
                    new_grid[r + 1, c] = grid[r, c]
                    new_grid[r, c] = 4
        return new_grid
    elif action == 3:
        # Toggle cells in specific rows (45-49, 61-62)
        new_grid = grid.copy()
        rows = [45, 46, 47, 48, 49, 61, 62]
        for r in rows:
            for c in range(grid.shape[1]):
                if grid[r, c] == 5:
                    new_grid[r, c] = 12
                elif grid[r, c] == 12:
                    new_grid[r, c] = 5
        return new_grid
    elif action == 6:
        # Click action (no change in observed data)
        return grid
    else:
        # Default: no change
        return grid

def is_level_complete(grid):
    # Check if grid matches the win state pattern
    # Win state has specific structure in rows 5-63
    # Check row 5
    if grid[5, 4] != 15 or grid[5, 5:39] != 3 or grid[5, 39:49] != 4:
        return False
    # Check row 6
    if grid[6, 4] != 15 or grid[6, 5:39] != 3 or grid[6, 39:49] != 4:
        return False
    # Check row 10
    if grid[10, 4] != 5 or grid[10, 5:49] != 3:
        return False
    # Check row 15
    if grid[15, 4] != 5 or grid[15, 5] != 3 or grid[15, 6:15] != 4 or grid[15, 15:20] != 3 or grid[15, 20:25] != 4 or grid[15, 25:30] != 3 or grid[15, 30:35] != 4 or grid[15, 35:40] != 3 or grid[15, 40:45] != 4 or grid[15, 45:49] != 3:
        return False
    # Check row 20
    if grid[20, 4] != 5 or grid[20, 5:15] != 3 or grid[20, 15:20] != 4 or grid[20, 20:25] != 3 or grid[20, 25:30] != 4 or grid[20, 30:35] != 3 or grid[20, 35:40] != 4 or grid[20, 40:45] != 3 or grid[20, 45:49] != 4:
        return False
    # Check row 25
    if grid[25, 4] != 10 or grid[25, 5:10] != 3 or grid[25, 10:15] != 4 or grid[25, 15:20] != 3 or grid[25, 20:25] != 4 or grid[25, 25:30] != 3 or grid[25, 30:35] != 4 or grid[25, 35:39] != 3 or grid[25, 39:44] != 4:
        return False
    # Check row 30
    if grid[30, 4] != 10 or grid[30, 5:10] != 3 or grid[30, 10:15] != 4 or grid[30, 15:20] != 3 or grid[30, 20:25] != 4 or grid[30, 25:30] != 3 or grid[30, 30:35] != 4 or grid[30, 35:39] != 3 or grid[30, 39:44] != 4:
        return False
    # Check row 35
    if grid[35, 4] != 10 or grid[35, 5:10] != 3 or grid[35, 10:15] != 4 or grid[35, 15:20] != 3 or grid[35, 20:25] != 4 or grid[35, 25:30] != 3 or grid[35, 30:35] != 4 or grid[35, 35:39] != 3 or grid[35, 39:44] != 4:
        return False
    # Check row 40
    if grid[40, 4] != 8 or grid[40, 5:8] != 3 or grid[40, 8:15] != 4 or grid[40, 15:20] != 3 or grid[40, 20:25] != 4 or grid[40, 25:30] != 3 or grid[40, 30:35] != 4 or grid[40, 35:39] != 3 or grid[40, 39:44] != 4:
        return False
    # Check row 45
    if grid[45, 4] != 8 or grid[45, 5:8] != 3 or grid[45, 8:15] != 4 or grid[45, 15:20] != 3 or grid[45, 20:25] != 4 or grid[45, 25:30] != 3 or grid[45, 30:35] != 4 or grid[45, 35:39] != 3 or grid[45, 39:44] != 4:
        return False
    # Check row 50
    if grid[50, 4] != 35 or grid[50, 5:39] != 3 or grid[50, 39:44] != 4:
        return False
    # Check row 51
    if grid[51, 4] != 35 or grid[51, 5:39] != 3 or grid[51, 39:44] != 4:
        return False
    # Check row 52
    if grid[52, 4] != 39 or grid[52, 5:39] != 3 or grid[52, 39:44] != 4:
        return False
    # Check row 53
    if grid[53, 4] != 1 or grid[53, 5:10] != 5 or grid[53, 10:28] != 4 or grid[53, 28:39] != 3 or grid[53, 39:44] != 4:
        return False
    # Check row 54
    if grid[54, 4] != 1 or grid[54, 5:10] != 5 or grid[54, 10:28] != 4 or grid[54, 28:39] != 3 or grid[54, 39:44] != 4:
        return False
    # Check row 55
    if grid[55, 4] != 1 or grid[55, 5:7] != 5 or grid[55, 7:12] != 9 or grid[55, 12:14] != 5 or grid[55, 14:44] != 4:
        return False
    # Check row 56
    if grid[56, 4] != 1 or grid[56, 5:7] != 5 or grid[56, 7:12] != 9 or grid[56, 12:14] != 5 or grid[56, 14:44] != 4:
        return False
    # Check row 57
    if grid[57, 4] != 1 or grid[57, 5:6] != 5 or grid[57, 6:12] != 9 or grid[57, 12:14] != 5 or grid[57, 14:44] != 4:
        return False
    # Check row 58
    if grid[58, 4] != 1 or grid[58, 5:6] != 5 or grid[58, 6:12] != 9 or grid[58, 12:14] != 5 or grid[58, 14:44] != 4:
        return False
    # Check row 59
    if grid[59, 4] != 1 or grid[59, 5:7] != 5 or grid[59, 7:9] != 9 or grid[59, 9:11] != 5 or grid[59, 11:13] != 9 or grid[59, 13:15] != 5 or grid[59, 15:44] != 4:
        return False
    # Check row 60
    if grid[60, 4] != 1 or grid[60, 5:7] != 5 or grid[60, 7:9] != 9 or grid[60, 9:11] != 5 or grid[60, 11:13] != 9 or grid[60, 13:15] != 5 or grid[60, 15:44] != 4 or grid[60, 44:46] != 5:
        return False
    # Check row 61
    if grid[61, 4] != 1 or grid[61, 5:10] != 5 or grid[61, 10:11] != 4 or grid[61, 11:12] != 5 or grid[61, 12:13] != 3 or grid[61, 13:54] != 11 or grid[61, 54:55] != 5 or grid[61, 55:56] != 8 or grid[61, 56:57] != 5 or grid[61, 57:58] != 8 or grid[61, 58:59] != 5 or grid[61, 59:60] != 8 or grid[61, 60:61] != 5 or grid[61, 61:63] != 8:
        return False
    # Check row 62
    if grid[62, 4] != 1 or grid[62, 5:10] != 5 or grid[62, 10:11] != 4 or grid[62, 11:12] != 5 or grid[62, 12:13] != 3 or grid[62, 13:54] != 11 or grid[62, 54:55] != 5 or grid[62, 55:56] != 8 or grid[62, 56:57] != 5 or grid[62, 57:58] != 8 or grid[62, 58:59] != 5 or grid[62, 59:60] != 8 or grid[62, 60:61] != 5 or grid[62, 61:63] != 8:
        return False
    # Check row 63
    if grid[63, 4] != 12 or grid[63, 5:63] != 5:
        return False
    return True