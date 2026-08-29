import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action == 0:
        # progress bar in column 0: fill next 5 cells (top-down) with color 5
        col0 = g[:, 0]
        # find first run of 5 cells that are not yet 5, starting from top
        i = 0
        H = g.shape[0]
        while i < H:
            if col0[i] != 5:
                # fill up to 5 cells from here that are not 5
                cnt = 0
                j = i
                while j < H and cnt < 5:
                    if col0[j] != 5:
                        g[j, 0] = 5
                        cnt += 1
                    j += 1
                break
            i += 1
        # color flow: for each 4x4 block, take the color of the block to its left
        # (same row band); if none, take the block above; if none, keep.
        # Detect block grid: blocks are 4x4 aligned to rows/cols that are multiples of 4 offset.
        # We'll scan for 4x4 uniform regions that are "blocks" (not background 4).
        H, W = g.shape
        # find block top-left positions: rows where a 4x4 uniform non-4 region starts
        # Use the known structure: blocks start at rows 19,25,31,37,43 and cols 12,18,24,30,36,42,48
        # Generalize: detect via uniform 4x4.
        def is_block(r, c):
            if r + 4 > H or c + 4 > W:
                return False
            sub = g[r:r+4, c:c+4]
            return np.all(sub == sub[0, 0]) and sub[0, 0] != 4
        # collect block positions
        blocks = []
        for r in range(0, H - 3):
            for c in range(0, W - 3):
                if is_block(r, c):
                    blocks.append((r, c))
        # dedupe: a block top-left is the min r,c of its 4x4; avoid duplicates by only keeping
        # those where the cell above-left are not part of same block
        # Simpler: keep positions where (r,c) is top-left: g[r-1,c]!=g[r,c] or r==0 and g[r,c-1]!=g[r,c] or c==0
        def is_topleft(r, c):
            if r > 0 and g[r-1, c] == g[r, c]:
                return False
            if c > 0 and g[r, c-1] == g[r, c]:
                return False
            return True
        blocks = [(r, c) for (r, c) in blocks if is_topleft(r, c)]
        # build map
        bmap = {}
        for (r, c) in blocks:
            bmap[(r, c)] = g[r, c]
        # for each block, new color = left neighbor's old color, else above neighbor's old color, else keep
        newvals = {}
        for (r, c) in blocks:
            left = bmap.get((r, c - 6))
            above = bmap.get((r - 6, c))
            if left is not None:
                newvals[(r, c)] = left
            elif above is not None:
                newvals[(r, c)] = above
            else:
                newvals[(r, c)] = bmap[(r, c)]
        for (r, c), v in newvals.items():
            g[r:r+4, c:c+4] = v
    return g

def is_level_complete(grid):
    # win when the whole column 0 progress bar is filled with 5
    return bool(np.all(grid[:, 0] == 5))
