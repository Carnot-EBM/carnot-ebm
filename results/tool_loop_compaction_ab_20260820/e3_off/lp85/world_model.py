import numpy as np

def _find_bands(g):
    H, W = g.shape
    # find rows that contain a 4x4 uniform non-4 block
    row_has = []
    for r in range(H - 3):
        found = False
        for c in range(W - 3):
            blk = g[r:r+4, c:c+4]
            if blk[0,0] != 4 and np.all(blk == blk[0,0]):
                found = True
                break
        row_has.append(found)
    # group consecutive found rows into bands of 4
    bands = []
    r = 0
    while r < len(row_has):
        if row_has[r]:
            start = r
            while r < len(row_has) and row_has[r]:
                r += 1
            bands.append(start)
        else:
            r += 1
    return bands

def _tile_cols(g, r0):
    W = g.shape[1]
    cols = []
    for c in range(W - 3):
        blk = g[r0:r0+4, c:c+4]
        if blk[0,0] != 4 and np.all(blk == blk[0,0]):
            cols.append(c)
    return cols

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action == 0:
        H, W = g.shape
        bands = _find_bands(g)
        for r0 in bands:
            cols = _tile_cols(g, r0)
            if len(cols) < 2:
                continue
            vals = [int(g[r0, c]) for c in cols]
            # left shift by 1, inject new value at right end
            newvals = vals[1:] + [vals[0]]
            for c, v in zip(cols, newvals):
                g[r0:r0+4, c:c+4] = v
        # marker in column 0: contiguous run of 5s moves down by 5
        col0 = g[:, 0]
        idx = np.where(col0 == 5)[0]
        if len(idx) > 0:
            blocks = []
            start = prev = idx[0]
            for i in idx[1:]:
                if i == prev + 1:
                    prev = i
                else:
                    blocks.append((start, prev)); start = prev = i
            blocks.append((start, prev))
            s, e = blocks[0]
            g[s:e+1, 0] = 14
            if e + 5 < H:
                g[s+5:e+6, 0] = 5
    return g

def is_level_complete(grid):
    return False
