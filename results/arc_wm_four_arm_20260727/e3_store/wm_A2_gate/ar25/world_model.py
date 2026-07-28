import numpy as np

def engine(grid, action, data):
    if action == 2:
        return apply_action_2(grid)
    elif action == 3:
        return apply_action_3(grid)
    elif action == 4:
        return apply_action_4(grid)
    elif action == 7:
        return grid
    elif action == 6:
        return grid
    else:
        return grid

def apply_action_2(grid):
    result = grid.copy()
    rows = grid.shape[0]
    cols = grid.shape[1]
    # Apply changes to rows 0-29
    for r in range(30):
        if r == 0:
            result[0, 63] = 5
        elif r == 15:
            result[15, 18:36] = 11
            result[15, 36:45] = 9
        elif r == 16:
            result[16, 18:36] = 11
            result[16, 36:45] = 9
        elif r == 17:
            result[17, 18:36] = 11
            result[17, 36:45] = 9
        elif r == 18:
            result[18, 18:24] = 5
            result[18, 39:45] = 9
        elif r == 19:
            result[19, 18:24] = 5
            result[19, 24:25] = 0
            result[19, 25:27] = 5
            result[19, 39:45] = 9
        elif r == 20:
            result[20, 18:24] = 5
            result[20, 39:45] = 9
        elif r == 24:
            result[24, 24:27] = 5
            result[24, 36:39] = 9
        elif r == 25:
            result[25, 24:25] = 5
            result[25, 25:26] = 0
            result[25, 26:27] = 5
            result[25, 36:39] = 9
        elif r == 26:
            result[26, 24:27] = 5
            result[26, 36:39] = 9
    return result

def apply_action_3(grid):
    result = grid.copy()
    rows = grid.shape[0]
    cols = grid.shape[1]
    # Apply changes to rows 1-29
    for r in range(30):
        if r == 1:
            result[1, 63] = 5
        elif r == 18:
            result[18, 15:18] = 5
            result[18, 24:27] = 11
            result[18, 36:39] = 11
            result[18, 45:48] = 9
        elif r == 19:
            result[19, 15:18] = 5
            result[19, 24:25] = 0
            result[19, 25:26] = 5
            result[19, 36:39] = 11
            result[19, 45:48] = 9
        elif r == 20:
            result[20, 15:18] = 5
            result[20, 24:27] = 11
            result[20, 36:39] = 11
            result[20, 45:48] = 9
        elif r == 21:
            result[21, 21:24] = 5
            result[21, 24:27] = 11
            result[21, 36:39] = 11
            result[21, 45:48] = 9
        elif r == 22:
            result[22, 21:24] = 5
            result[22, 24:25] = 0
            result[22, 25:26] = 5
            result[22, 26:27] = 11
            result[22, 36:39] = 11
            result[22, 45:48] = 9
        elif r == 23:
            result[23, 21:24] = 5
            result[23, 24:27] = 11
            result[23, 36:39] = 11
            result[23, 45:48] = 9
        elif r == 24:
            result[24, 21:24] = 5
            result[24, 24:27] = 11
            result[24, 36:39] = 11
            result[24, 45:48] = 9
        elif r == 25:
            result[25, 21:24] = 5
            result[25, 24:25] = 0
            result[25, 25:26] = 5
            result[25, 26:27] = 11
            result[25, 36:39] = 11
            result[25, 45:48] = 9
        elif r == 26:
            result[26, 21:24] = 5
            result[26, 24:27] = 11
            result[26, 36:39] = 1
            result[26, 45:48] = 9
    return result

def apply_action_4(grid):
    result = grid.copy()
    rows = grid.shape[0]
    cols = grid.shape[1]
    # Apply changes to rows 3-29
    for r in range(30):
        if r == 3:
            result[3, 63] = 5
        elif r == 21:
            result[21, 15:18] = 11
            result[21, 24:27] = 5
            result[21, 36:39] = 9
            result[21, 45:48] = 11
        elif r == 22:
            result[22, 15:18] = 11
            result[22, 24:25] = 5
            result[22, 25:26] = 0
            result[22, 26:27] = 5
            result[22, 36:39] = 9
            result[22, 45:48] = 11
        elif r == 23:
            result[23, 15:18] = 11
            result[23, 24:27] = 5
            result[23, 36:39] = 9
            result[23, 45:48] = 11
        elif r == 24:
            result[24, 21:24] = 11
            result[24, 24:27] = 5
            result[24, 36:39] = 9
            result[24, 45:48] = 11
        elif r == 25:
            result[25, 21:24] = 11
            result[25, 24:25] = 5
            result[25, 25:26] = 0
            result[25, 26:27] = 5
            result[25, 36:39] = 9
            result[25, 45:48] = 11
        elif r == 26:
            result[26, 21:24] = 11
            result[26, 24:27] = 5
            result[26, 36:39] = 9
            result[26, 45:48] = 11
        elif r == 27:
            result[27, 21:24] = 11
            result[27, 24:27] = 5
            result[27, 36:39] = 9
            result[27, 45:48] = 11
        elif r == 28:
            result[28, 21:24] = 11
            result[28, 24:25] = 5
            result[28, 25:26] = 0
            result[28, 26:27] = 5
            result[28, 36:39] = 9
            result[28, 45:48] = 11
        elif r == 29:
            result[29, 21:24] = 11
            result[29, 24:27] = 5
            result[29, 36:39] = 9
            result[29, 45:48] = 11
    return result

def is_level_complete(grid):
    return False