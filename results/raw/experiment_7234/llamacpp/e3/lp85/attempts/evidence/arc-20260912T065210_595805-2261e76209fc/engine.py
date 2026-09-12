import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action != 6 or data is None:
        return g
    px, py = int(data.get('x', 0)), int(data.get('y', 0))
    H, W = g.shape
    if py >= H or px >= W:
        return g
    # Only the color-8 "button" triggers a rotation of all 4x4 blocks.
    if g[py, px] != 8:
        return g

    # Collect all 4x4 uniform blocks (non-background colors).
    bg = {3, 4, 14, 5}
    blocks = []
    for r in range(0, H - 3):
        for c in range(0, W - 3):
            v = g[r, c]
            if v in bg:
                continue
            # check 4x4 uniform
            sub = g[r:r+4, c:c+4]
            if sub.shape == (4, 4) and np.all(sub == v):
                blocks.append((r, c, v))

    # Group by row (top row of block)
    from collections import defaultdict
    row_blocks = defaultdict(list)
    for (r, c, v) in blocks:
        row_blocks[r].append((c, v))

    # For each row, shift colors left by one (circular).
    for r, blist in row_blocks.items():
        blist.sort()
        n = len(blist)
        if n > 1:
            for i, (c, v) in enumerate(blist):
                newv = blist[(i + 1) % n][1]
                g[r:r+4, c:c+4] = newv

    # Left column (col 0): top rows change 14 -> 5 (5 cells observed).
    # Generalize: change a run of 14 at the top of col 0 to 5.
    cnt = 0
    for r in range(H):
        if g[r, 0] == 14:
            g[r, 0] = 5
            cnt += 1
            if cnt >= 5:
                break
        else:
            break
    return g

def is_level_complete(grid):
    return False
