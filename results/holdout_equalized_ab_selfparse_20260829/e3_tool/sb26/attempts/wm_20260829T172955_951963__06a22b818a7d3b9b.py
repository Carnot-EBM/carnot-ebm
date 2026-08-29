import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    if action != 6 or data is None:
        return g
    px, py = int(data['x']), int(data['y'])
    c, r = px, py
    H, W = g.shape
    if not (0 <= r < H and 0 <= c < W):
        return g
    v = int(g[r, c])
    if v == 0 or v == 4 or v == 5 or v == 8:
        return g
    # dig a 1-wide moat of 0 around the clicked block's bounding box
    mask = (g == v)
    # restrict to the connected component containing (r,c)
    from collections import deque
    seen = np.zeros_like(mask, dtype=bool)
    dq = deque([(r, c)])
    seen[r, c] = True
    minr = maxr = r
    minc = maxc = c
    while dq:
        rr, cc = dq.popleft()
        minr = min(minr, rr); maxr = max(maxr, rr)
        minc = min(minc, cc); maxc = max(maxc, cc)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and mask[nr, nc]:
                    seen[nr, nc] = True
                    dq.append((nr, nc))
    # moat = bbox expanded by 1, border cells that are background(4) become 0
    for rr in range(max(0, minr - 1), min(H, maxr + 2)):
        for cc in range(max(0, minc - 1), min(W, maxc + 2)):
            if rr in (minr - 1, maxr + 1) or cc in (minc - 1, maxc + 1):
                if g[rr, cc] == 4:
                    g[rr, cc] = 0
    return g

def is_level_complete(grid):
    return False
