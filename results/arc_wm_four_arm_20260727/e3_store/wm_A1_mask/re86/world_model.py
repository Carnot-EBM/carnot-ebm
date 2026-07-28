import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    if action == 2:
        if data is None:
            return grid
        # Action 2 is a click that toggles a block of cells
        px, py = data['x'], data['y']
        # The block is 3x3 centered at (px, py)
        r1, r2 = max(0, py - 1), min(H - 1, py + 1)
        c1, c2 = max(0, px - 1), min(W - 1, px + 1)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if grid[r, c] == 3:
                    grid[r, c] = 7
                else:
                    grid[r, c] = 3
        return grid
    elif action == 3:
        if data is None:
            return grid
        # Action 3 is a click that toggles a block of cells
        px, py = data['x'], data['y']
        # The block is 3x3 centered at (px, py)
        r1, r2 = max(0, py - 1), min(H - 1, py + 1)
        c1, c2 = max(0, px - 1), min(W - 1, px + 1)
        for r in range(r1, r2 + 1):
            for c in range(c1, c2 + 1):
                if grid[r, c] == 3:
                    grid[r, c] = 7
                else:
                    grid[r, c] = 3
        return grid
    else:
        return grid

def is_level_complete(grid):
    # Check if there are any 3s left in the grid
    return not np.any(grid == 3)