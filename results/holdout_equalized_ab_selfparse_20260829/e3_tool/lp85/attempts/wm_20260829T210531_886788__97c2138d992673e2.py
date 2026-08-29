import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    if action == 0:
        # left column (col 0) top 5 cells become 5
        for r in range(0, min(5, H)):
            g[r, 0] = 5
        # find 4x4 colored blocks (color not in {3,4,14}) and shift colors left
        # detect block grid: columns and rows where a 4x4 block sits
        # scan for 4x4 regions of a single non-background color
        bg = {3, 4, 14}
        # collect candidate block top-left positions
        blocks = []
        for r in range(0, H - 3):
            for c in range(0, W - 3):
                sub = g[r:r+4, c:c+4]
                if sub.shape == (4, 4):
                    vals = set(int(x) for x in sub.flatten())
                    if len(vals) == 1 and vals.pop() not in bg:
                        blocks.append((r, c))
        # group into rows/cols
        if blocks:
            rows = sorted(set(r for r, c in blocks))
            cols = sorted(set(c for r, c in blocks))
            # build map
            pos = {(r, c) for r, c in blocks}
            newg = g.copy()
            for ri, r in enumerate(rows):
                for ci, c in enumerate(cols):
                    if (r, c) not in pos:
                        continue
                    cur = int(g[r, c])
                    # source: block to the left in same row; if none, block above in same col
                    src = None
                    if ci > 0 and (r, cols[ci-1]) in pos:
                        src = int(g[r, cols[ci-1]])
                    elif ri > 0 and (rows[ri-1], c) in pos:
                        src = int(g[rows[ri-1], c])
                    if src is not None:
                        newg[r:r+4, c:c+4] = src
            g = newg
    return g

def is_level_complete(grid):
    return False
