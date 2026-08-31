import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action == 4:
        # Action 4: directional (right?). Hypothesis: erodes bottom row from right.
        # Find bottom row, change rightmost 2 cells of color 6 to 0.
        h, w = g.shape
        bottom = g[h-1]
        # Find rightmost contiguous run of 6s at the end
        count = 0
        for c in range(w-1, -1, -1):
            if bottom[c] == 6:
                count += 1
            else:
                break
        # Erase up to 2 cells from the right of that run
        erase = min(2, count)
        for i in range(erase):
            g[h-1, w-1-i] = 0
    return g

def is_level_complete(grid):
    # No win state observed; default False
    return False