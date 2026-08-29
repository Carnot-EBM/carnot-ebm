import numpy as np

def _find_player(grid):
    # player = 3x3 block of 9s (with a 4 marker inside)
    ys, xs = np.where(grid == 9)
    if len(ys) == 0:
        return None
    r0, r1 = int(ys.min()), int(ys.max())
    c0, c1 = int(xs.min()), int(xs.max())
    return (r0, c0, r1, c1)

def engine(grid, action, data):
    g = grid.astype(int).copy()
    p = _find_player(g)
    if p is None:
        return g
    r0, c0, r1, c1 = p
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    # find facing marker (4)
    sub = g[r0:r1+1, c0:c1+1]
    fr, fc = np.where(sub == 4)
    if len(fr) > 0:
        fr, fc = int(fr[0]), int(fc[0])
    else:
        fr, fc = h//2, w//2
    # clear old player block
    g[r0:r1+1, c0:c1+1] = 0
    # determine move
    step = 6
    dr, dc = 0, 0
    if action == 1:
        dr = -step
    elif action == 2:
        dr = step
    elif action == 3:
        dc = -step
    elif action == 4:
        dc = step
    nr0 = r0 + dr
    nc0 = c0 + dc
    H, W = g.shape
    # clamp
    if nr0 < 0:
        nr0 = 0
    if nc0 < 0:
        nc0 = 0
    if nr0 + h > H:
        nr0 = H - h
    if nc0 + w > W:
        nc0 = W - w
    # new facing
    if action == 1:
        nfr, nfc = 0, w//2
    elif action == 2:
        nfr, nfc = h-1, w//2
    elif action == 3:
        nfr, nfc = h//2, 0
    elif action == 4:
        nfr, nfc = h//2, w-1
    else:
        nfr, nfc = fr, fc
    # place new player
    g[nr0:nr0+h, nc0:nc0+w] = 9
    g[nr0+nfr, nc0+nfc] = 4
    return g

def is_level_complete(grid):
    return False
