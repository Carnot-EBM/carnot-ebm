import numpy as np

def _convert_block(grid, r, c):
    # convert the 6x6 block of 9s whose top-left is (r,c) to 8
    for i in range(6):
        for j in range(6):
            rr, cc = r + i, c + j
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                if grid[rr, cc] == 9:
                    grid[rr, cc] = 8

def _advance_bar(grid):
    # fill the bottom row (last row) with 11 from the right, 2 cells per click
    H, W = grid.shape
    row = H - 1
    # find the leftmost 11 currently in the bar (the filled region is a suffix)
    # count how many 11s are already in the suffix
    filled = 0
    for cc in range(W - 1, -1, -1):
        if grid[row, cc] == 11:
            filled += 1
        else:
            break
    # add 2 more cells to the left of the filled suffix
    new_filled = filled + 2
    # set the suffix of length new_filled to 11
    for cc in range(W - new_filled, W):
        if 0 <= cc < W:
            grid[row, cc] = 11

def engine(grid, action, data):
    g = grid.copy()
    if action == 6 and data is not None:
        x = int(data.get('x', 0))
        y = int(data.get('y', 0))
        # click at top-left of a 6x6 block of 9s
        _convert_block(g, y, x)
        _advance_bar(g)
    return g

def is_level_complete(grid):
    # level complete when no 6x6 block of 9s remains in the active panel
    # (all targets converted to 8). Simple proxy: no 9-colored 6x6 blocks at all.
    H, W = grid.shape
    for r in range(H - 5):
        for c in range(W - 5):
            block = grid[r:r+6, c:c+6]
            if np.all(block == 9):
                return False
    return True
