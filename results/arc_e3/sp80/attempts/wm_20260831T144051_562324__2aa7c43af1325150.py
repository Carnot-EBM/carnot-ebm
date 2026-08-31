import numpy as np

def _move_block(g, color, dr, dc, bg):
    mask = (g == color)
    if not mask.any():
        return g
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    H, W = g.shape
    nr0, nr1 = r0 + dr, r1 + dr
    nc0, nc1 = c0 + dc, c1 + dc
    if nc0 < 0: nc0 = 0
    if nc1 >= W: nc1 = W - 1
    if nr0 < 0: nr0 = 0
    if nr1 >= H: nr1 = H - 1
    g[r0:r1 + 1, c0:c1 + 1] = bg
    g[nr0:nr1 + 1, nc0:nc1 + 1] = color
    return g

def _decrement_bar(g, bar_color=14, empty=0):
    # The bar is a run of bar_color cells with empty cells to its right (left-anchored).
    # Find the row containing the bar (the row with the most bar_color cells).
    H, W = g.shape
    counts = (g == bar_color).sum(axis=1)
    if counts.max() == 0:
        return g
    r = int(np.argmax(counts))
    row = g[r]
    # leftmost empty cell that has a bar_color cell to its left
    for c in range(W):
        if row[c] == empty and c > 0 and row[c - 1] == bar_color:
            row[c] = bar_color  # placeholder, will set below
            g[r, c] = empty
            return g
    # fallback: leftmost bar cell
    idx = np.where(row == bar_color)[0]
    if len(idx):
        g[r, idx[0]] = empty
    return g

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    # background color: most common color in the grid
    vals, cnts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(cnts)])
    # Determine movement by action (3=left,4=right,1=up,2=down); step = block height
    dr, dc = 0, 0
    if action == 3:
        dc = -4
    elif action == 4:
        dc = 4
    elif action == 1:
        dr = -4
    elif action == 2:
        dr = 4
    if dr or dc:
        g = _move_block(g, 9, dr, dc, bg)
    # HUD progress bar decrements by 1 per action
    g = _decrement_bar(g, 14, 0)
    return g

def is_level_complete(grid):
    return False
