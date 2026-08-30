import numpy as np

def engine(grid, action, data):
    g = grid.astype(int).copy()
    if action != 6:
        return g
    # Find the "face" object: a 3-row-tall block where the leftmost column is all 3,
    # and the rest is 14 with a single 13 "eye" in the middle row.
    # Strategy: find all 3-cells that are part of a vertical 3-run of length 3,
    # then check the block to the right.
    H, W = g.shape
    best = None
    for r in range(H-2):
        for c in range(W):
            if g[r,c]==3 and g[r+1,c]==3 and g[r+2,c]==3:
                # check right side is 14 (with possible 13)
                # find extent of 14 to the right
                c2 = c+1
                while c2 < W and g[r,c2] in (14,13):
                    c2 += 1
                # block is cols c..c2-1, rows r..r+2
                # verify it's a solid 14 block (except the 3 col and one 13)
                ok = True
                n13 = 0
                for rr in range(r, r+3):
                    for cc in range(c+1, c2):
                        if g[rr,cc] not in (14,13):
                            ok = False
                        if g[rr,cc]==13:
                            n13 += 1
                if ok and n13==1:
                    best = (r, c, c2-1)
                    break
        if best:
            break
    if best is None:
        return g
    r0, c0, c1 = best
    # extract object pattern (3 col + 14/13 body)
    obj = g[r0:r0+3, c0:c1+1].copy()
    # clear original
    for r in range(r0, r0+3):
        for c in range(c0, c1+1):
            if g[r,c] in (3,14,13):
                g[r,c] = 5
    # move right by 3
    dc = 3
    nc0 = c0 + dc
    if nc0 + (c1-c0+1) <= W:
        for i in range(3):
            for j in range(c1-c0+1):
                v = obj[i,j]
                if v in (3,14,13):
                    g[r0+i, nc0+j] = v
    # bottom-right counter: rightmost 4 in last row shifts left by 1
    last = H-1
    col4 = None
    for c in range(W-1, -1, -1):
        if g[last,c]==4:
            col4 = c
            break
    if col4 is not None and col4>0:
        g[last,col4]=5
        g[last,col4-1]=4
    return g

def is_level_complete(grid):
    return False
