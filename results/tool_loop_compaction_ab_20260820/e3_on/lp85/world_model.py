import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    # 1) left column (col 0) fills with color 5, 5 rows per step, top-down
    filled = int(np.sum(g[:, 0] == 5))
    for r in range(filled, min(filled + 5, g.shape[0])):
        g[r, 0] = 5
    # 2) 4x4 blocks: shift colors left (each block takes color of block to its right)
    #    find block top-left corners: 4x4 regions of a non-4, non-14, non-3, non-5, non-8 color
    H, W = g.shape
    block_colors = {}
    for r in range(0, H - 3):
        for c in range(0, W - 3):
            v = g[r, c]
            if v in (1, 2, 9, 10, 11, 15):
                # check it's a full 4x4 block of same color
                if np.all(g[r:r+4, c:c+4] == v):
                    block_colors[(r, c)] = v
    # group by row band (same r)
    from collections import defaultdict
    rows = defaultdict(list)
    for (r, c), v in block_colors.items():
        rows[r].append((c, v))
    for r, lst in rows.items():
        lst.sort()
        for i, (c, v) in enumerate(lst):
            if i + 1 < len(lst):
                newv = lst[i+1][1]
            else:
                newv = 15  # rightmost takes 15 (guess)
            g[r:r+4, c:c+4] = newv
    return g

def is_level_complete(grid):
    return False
