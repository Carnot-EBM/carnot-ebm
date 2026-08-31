import numpy as np

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    H, W = g.shape
    # find the 10-colored objects (5x5 blocks)
    mask = (g == 10)
    # connected components (4-conn)
    seen = np.zeros_like(mask, dtype=bool)
    objs = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                # BFS
                stack = [(r, c)]
                comp = []
                seen[r, c] = True
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
                objs.append(comp)
    # keep only 5x5 blocks
    blocks = []
    for comp in objs:
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        if (max(rs)-min(rs)+1 == 5) and (max(cs)-min(cs)+1 == 5) and len(comp) == 25:
            blocks.append((min(rs), min(cs)))
    # sort by column
    blocks.sort(key=lambda b: b[1])
    if len(blocks) >= 2:
        L = blocks[0]
        R = blocks[-1]
        dr = dc = 0
        if action == 1:
            dr = -1
        elif action == 2:
            dr = 1
        elif action == 3:
            dc = -1
        elif action == 4:
            dc = 1
        step = 5
        # move L
        for (r0, c0) in [(L, (dr, dc)), (R, (dr, -dc))]:
            r0, c0 = r0
            drr, dcc = (dr, dc) if r0 == L else (dr, -dc)
            nr0 = r0 + drr*step
            nc0 = c0 + dcc*step
            if 0 <= nr0 and nr0+5 <= H and 0 <= nc0 and nc0+5 <= W:
                # clear old
                g[r0:r0+5, c0:c0+5] = 5
                # place new
                g[nr0:nr0+5, nc0:nc0+5] = 10
    return g

def is_level_complete(grid):
    return False
