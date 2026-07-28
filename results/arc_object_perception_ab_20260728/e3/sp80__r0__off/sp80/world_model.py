import numpy as np

def engine(grid, action, data):
    if action == 4:
        if data is None:
            return grid
        px, py = data['x'], data['y']
        h, w = grid.shape
        new_grid = grid.copy()
        for dy in range(1, 5):
            for dx in range(1, 5):
                if 0 <= py + dy < h and 0 <= px + dx < w:
                    if new_grid[py + dy, px + dx] == 12:
                        new_grid[py + dy, px + dx] = 0
        return new_grid

    if action == 5:
        h, w = grid.shape
        new_grid = grid.copy()
        # Fill column 0
        for r in range(h):
            if new_grid[r, 0] == 0:
                new_grid[r, 0] = 1
        # Fill column 12
        for r in range(h):
            if new_grid[r, 12] == 0:
                new_grid[r, 12] = 1
        # Fill column 28
        for r in range(h):
            if new_grid[r, 28] == 0:
                new_grid[r, 28] = 1
        # Fill column 44
        for r in range(h):
            if new_grid[r, 44] == 0:
                new_grid[r, 44] = 1
        # Fill column 60
        for r in range(h):
            if new_grid[r, 60] == 0:
                new_grid[r, 60] = 1
        # Fill column 62
        for r in range(h):
            if new_grid[r, 62] == 0:
                new_grid[r, 62] = 1
        # Fill column 63
        for r in range(h):
            if new_grid[r, 63] == 0:
                new_grid[r, 63] = 1
        # Fill column 64 (out of bounds, ignore)
        # Fill column 61
        for r in range(h):
            if new_grid[r, 61] == 0:
                new_grid[r, 61] = 1
        # Fill column 59
        for r in range(h):
            if new_grid[r, 59] == 0:
                new_grid[r, 59] = 1
        # Fill column 58
        for r in range(h):
            if new_grid[r, 58] == 0:
                new_grid[r, 58] = 1
        # Fill column 57
        for r in range(h):
            if new_grid[r, 57] == 0:
                new_grid[r, 57] = 1
        # Fill column 56
        for r in range(h):
            if new_grid[r, 56] == 0:
                new_grid[r, 56] = 1
        # Fill column 55
        for r in range(h):
            if new_grid[r, 55] == 0:
                new_grid[r, 55] = 1
        # Fill column 54
        for r in range(h):
            if new_grid[r, 54] == 0:
                new_grid[r, 54] = 1
        # Fill column 53
        for r in range(h):
            if new_grid[r, 53] == 0:
                new_grid[r, 53] = 1
        # Fill column 52
        for r in range(h):
            if new_grid[r, 52] == 0:
                new_grid[r, 52] = 1
        # Fill column 51
        for r in range(h):
            if new_grid[r, 51] == 0:
                new_grid[r, 51] = 1
        # Fill column 50
        for r in range(h):
            if new_grid[r, 50] == 0:
                new_grid[r, 50] = 1
        # Fill column 49
        for r in range(h):
            if new_grid[r, 49] == 0:
                new_grid[r, 49] = 1
        # Fill column 48
        for r in range(h):
            if new_grid[r, 48] == 0:
                new_grid[r, 48] = 1
        # Fill column 47
        for r in range(h):
            if new_grid[r, 47] == 0:
                new_grid[r, 47] = 1
        # Fill column 46
        for r in range(h):
            if new_grid[r, 46] == 0:
                new_grid[r, 46] = 1
        # Fill column 45
        for r in range(h):
            if new_grid[r, 45] == 0:
                new_grid[r, 45] = 1
        # Fill column 43
        for r in range(h):
            if new_grid[r, 43] == 0:
                new_grid[r, 43] = 1
        # Fill column 42
        for r in range(h):
            if new_grid[r, 42] == 0:
                new_grid[r, 42] = 1
        # Fill column 41
        for r in range(h):
            if new_grid[r, 41] == 0:
                new_grid[r, 41] = 1
        # Fill column 40
        for r in range(h):
            if new_grid[r, 40] == 0:
                new_grid[r, 40] = 1
        # Fill column 39
        for r in range(h):
            if new_grid[r, 39] == 0:
                new_grid[r, 39] = 1
        # Fill column 38
        for r in range(h):
            if new_grid[r, 38] == 0:
                new_grid[r, 38] = 1
        # Fill column 37
        for r in range(h):
            if new_grid[r, 37] == 0:
                new_grid[r, 37] = 1
        # Fill column 36
        for r in range(h):
            if new_grid[r, 36] == 0:
                new_grid[r, 36] = 1
        # Fill column 35
        for r in range(h):
            if new_grid[r, 35] == 0:
                new_grid[r, 35] = 1
        # Fill column 34
        for r in range(h):
            if new_grid[r, 34] == 0:
                new_grid[r, 34] = 1
        # Fill column 33
        for r in range(h):
            if new_grid[r, 33] == 0:
                new_grid[r, 33] == 1
        # Fill column 32
        for r in range(h):
            if new_grid[r, 32] == 0:
                new_grid[r, 32] == 1
        # Fill column 31
        for r in range(h):
            if new_grid[r, 31] == 0:
                new_grid[r, 31] == 1
        # Fill column 30
        for r in range(h):
            if new_grid[r, 30] == 0:
                new_grid[r, 30] == 1
        # Fill column 29
        for r in range(h):
            if new_grid[r, 29] == 0:
                new_grid[r, 29] == 1
        # Fill column 27
        for r in range(h):
            if new_grid[r, 27] == 0:
                new_grid[r, 27] == 1
        # Fill column 26
        for r in range(h):
            if new_grid[r, 26] == 0:
                new_grid[r, 26] == 1
        # Fill column 25
        for r in range(h):
            if new_grid[r, 25] == 0:
                new_grid[r, 25] == 1
        # Fill column 24
        for r in range(h):
            if new_grid[r, 24] == 0:
                new_grid[r, 24] == 1
        # Fill column 23
        for r in range(h):
            if new_grid[r, 23] == 0:
                new_grid[r, 23] == 1
        # Fill column 22
        for r in range(h):
            if new_grid[r, 22] == 0:
                new_grid[r, 22] == 1
        # Fill column 21
        for r in range(h):
            if new_grid[r, 21] == 0:
                new_grid[r, 21] == 1
        # Fill column 20
        for r in range(h):
            if new_grid[r, 20] == 0:
                new_grid[r, 20] == 1
        # Fill column 19
        for r in range(h):
            if new_grid[r, 19] == 0:
                new_grid[r, 19] == 1
        # Fill column 18
        for r in range(h):
            if new_grid[r, 18] == 0:
                new_grid[r, 18] == 1
        # Fill column 17
        for r in range(h):
            if new_grid[r, 17] == 0:
                new_grid[r, 17] == 1
        # Fill column 16
        for r in range(h):
            if new_grid[r, 16] == 0:
                new_grid[r, 16] == 1
        # Fill column 15
        for r in range(h):
            if new_grid[r, 15] == 0:
                new_grid[r, 15] == 1
        # Fill column 14
        for r in range(h):
            if new_grid[r, 14] == 0:
                new_grid[r, 14] == 1
        # Fill column 13
        for r in range(h):
            if new_grid[r, 13] == 0:
                new_grid[r, 13] == 1
        # Fill column 11
        for r in range(h):
            if new_grid[r, 11] == 0:
                new_grid[r, 11] == 1
        # Fill column 10
        for r in range(h):
            if new_grid[r, 10] == 0:
                new_grid[r, 10] == 1
        # Fill column 9
        for r in range(h):
            if new_grid[r, 9] == 0:
                new_grid[r, 9] == 1
        # Fill column 8
        for r in range(h):
            if new_grid[r, 8] == 0:
                new_grid[r, 8] == 1
        # Fill column 7
        for r in range(h):
            if new_grid[r, 7] == 0:
                new_grid[r, 7] == 1
        # Fill column 6
        for r in range(h):
            if new_grid[r, 6] == 0:
                new_grid[r, 6] == 1
        # Fill column 5
        for r in range(h):
            if new_grid[r, 5] == 0:
                new_grid[r, 5] == 1
        # Fill column 4
        for r in range(h):
            if new_grid[r, 4] == 0:
                new_grid[r, 4] == 1
        # Fill column 3
        for r in range(h):
            if new_grid[r, 3] == 0:
                new_grid[r, 3] == 1
        # Fill column 2
        for r in range(h):
            if new_grid[r, 2] == 0:
                new_grid[r, 2] == 1
        # Fill column 1
        for r in range(h):
            if new_grid[r, 1] == 0:
                new_grid[r, 1] == 1
        # Fill column 0 (already done)
        return new_grid

    return grid

def is_level_complete(grid):
    h, w = grid.shape
    # Check if all cells are filled with 1
    if np.all(grid == 1):
        return True
    # Check if all cells are filled with 12
    if np.all(grid == 12):
        return True
    # Check if all cells are filled with 14
    if np.all(grid == 14):
        return True
    # Check if all cells are filled with 11
    if np.all(grid == 11):
        return True
    # Check if all cells are filled with 4
    if np.all(grid == 4):
        return True
    # Check if all cells are filled with 6
    if np.all(grid == 6):
        return True
    # Check if all cells are filled with 9
    if np.all(grid == 9):
        return True
    # Check if all cells are filled with 12
    if np.all(grid == 12):
        return True
    # Check if all cells are filled with 14
    if np.all(grid == 14):
        return True
    # Check if all cells are filled with 11
    if np.all(grid == 11):
        return True
    # Check if all cells are filled with 4
    if np.all(grid == 4):
        return True
    # Check if all cells are filled with 6
    if np.all(grid == 6):
        return True
    # Check if all cells are filled with 9
    if np.all(grid == 9):
        return True
    return False