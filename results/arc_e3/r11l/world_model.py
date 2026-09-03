import numpy as np

def _find_6(g):
    idx = np.argwhere(g == 6)
    if len(idx) == 0:
        return None
    return int(idx[0][0]), int(idx[0][1])

def _find_15_block(g):
    idx = np.argwhere(g == 15)
    if len(idx) == 0:
        return None
    r0, c0 = idx.min(axis=0)
    r1, c1 = idx.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)

def _find_1s(g):
    idx = np.argwhere(g == 1)
    return [(int(r), int(c)) for r, c in idx]

def _find_3s(g):
    idx = np.argwhere(g == 3)
    return [(int(r), int(c)) for r, c in idx]

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    if action != 6 or data is None:
        return g
    r = int(data.get('y', 0))
    c = int(data.get('x', 0))
    if not (0 <= r < g.shape[0] and 0 <= c < g.shape[1]):
        return g

    # counter at (0,0): cycles 0..5
    cur = int(g[0, 0])
    g[0, 0] = (cur + 1) % 6

    six = _find_6(g)
    if six is None:
        return g
    sr, sc = six
    if (sr, sc) != (r, c):
        return g

    # 15 block
    blk = _find_15_block(g)
    if blk is None:
        return g
    br0, bc0, br1, bc1 = blk
    bh = br1 - br0 + 1
    bw = bc1 - bc0 + 1

    # 1s
    ones = _find_1s(g)
    # 3s
    threes = _find_3s(g)

    # move 15 block by (+4, +5)
    new_br0, new_bc0 = br0 + 4, bc0 + 5
    new_br1, new_bc1 = br1 + 4, bc1 + 5

    # clear old block region
    g[br0:br1+1, bc0:bc1+1] = 5
    # place new block
    if new_br1 < g.shape[0] and new_bc1 < g.shape[1]:
        g[new_br0:new_br1+1, new_bc0:new_bc1+1] = 15
        # 6 at center of new block
        nr = new_br0 + bh // 2
        nc = new_bc0 + bw // 2
        g[nr, nc] = 6

    # move 1s by (+4, +5)
    for (rr, cc) in ones:
        nr, nc = rr + 4, cc + 5
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            g[rr, cc] = 5
            g[nr, nc] = 1

    # move 3s by (+4, +5)
    for (rr, cc) in threes:
        nr, nc = rr + 4, cc + 5
        if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
            g[rr, cc] = 5
            g[nr, nc] = 3

    # diamond at old 6 position
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            rr, cc = sr + dr, sc + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                d = abs(dr) + abs(dc)
                if d == 0:
                    g[rr, cc] = 15
                elif d <= 2:
                    g[rr, cc] = 0
                else:
                    g[rr, cc] = 5

    return g

def is_level_complete(grid):
    return False
