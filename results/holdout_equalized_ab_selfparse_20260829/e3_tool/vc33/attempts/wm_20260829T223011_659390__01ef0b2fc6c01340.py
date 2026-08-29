import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action != 6 or data is None:
        return g
    x = int(data.get('x', 0))
    y = int(data.get('y', 0))
    step = 4
    # Top half (rows 1..27): 3-region expands right by step (0->3)
    for r in range(1, 28):
        row = g[r]
        c = 0
        while c < W and row[c] == 3:
            c += 1
        edge = c
        for k in range(step):
            cc = edge + k
            if cc < W and row[cc] == 0:
                row[cc] = 3
    # Bottom half (rows 32..H-1): 3-region + 4/11 blocks shift left by step
    for r in range(32, H):
        row = g[r]
        c = 0
        while c < W and row[c] == 3:
            c += 1
        edge3 = c
        blocklen = 0
        cc = edge3
        while cc < W and row[cc] in (4, 11):
            blocklen += 1
            cc += 1
        if blocklen > 0:
            newvals = row[edge3:edge3+blocklen].copy()
            for k in range(blocklen + step):
                if edge3 + k < W:
                    row[edge3 + k] = 0
            for k in range(step):
                row[edge3 - step + k] = 3
            for k in range(blocklen):
                row[edge3 - step + k] = newvals[k]
        else:
            for k in range(step):
                cc = edge3 - 1 - k
                if cc >= 0 and row[cc] == 3:
                    row[cc] = 0
    # Top row: convert rightmost (n+1) 7s to 4s, where n = current count of 4s in row0
    row0 = g[0]
    n4 = int(np.sum(row0 == 4))
    to_convert = n4 + 1
    # find rightmost 7s and convert 'to_convert' of them
    c = W - 1
    converted = 0
    while c >= 0 and converted < to_convert:
        if row0[c] == 7:
            row0[c] = 4
            converted += 1
        c -= 1
    return g

def is_level_complete(grid):
    return False
