import numpy as np

def _find_block(g, r, c, color, size):
    H, W = g.shape
    for dr in range(-(size-1), 1):
        for dc in range(-(size-1), 1):
            r0, c0 = r + dr, c + dc
            if r0 < 0 or c0 < 0 or r0 + size > H or c0 + size > W:
                continue
            if np.all(g[r0:r0+size, c0:c0+size] == color):
                return (r0, c0)
    return None

def _ring_to_zero(g, r0, c0, size):
    H, W = g.shape
    for rr in range(r0-1, r0+size+1):
        for cc in range(c0-1, c0+size+1):
            if 0 <= rr < H and 0 <= cc < W:
                if rr == r0-1 or rr == r0+size or cc == c0-1 or cc == c0+size:
                    g[rr, cc] = 0

def _find_cleared_slot(g):
    H, W = g.shape
    for rr in range(H-5):
        for cc in range(W-5):
            if np.all(g[rr:rr+6, cc:cc+6] == 0):
                return (rr, cc)
    return None

def _top_color(g, c0):
    H, W = g.shape
    best, bestn = None, 0
    for col in (9, 14, 11, 15):
        n = 0
        for rr in range(0, min(11, H)):
            for cc in range(max(0, c0-2), min(W, c0+4)):
                if g[rr, cc] == col:
                    n += 1
        if n > bestn:
            bestn, best = n, col
    return best

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64)
    if action != 6 or data is None:
        return g
    x = int(data.get('x', 0)); y = int(data.get('y', 0))
    r, c = y, x
    H, W = g.shape
    if not (0 <= r < H and 0 <= c < W):
        return g
    v = int(g[r, c])
    if v in (9, 14, 11, 15):
        blk = _find_block(g, r, c, v, 4)
        if blk is not None:
            _ring_to_zero(g, blk[0], blk[1], 4)
            return g
    if v == 2:
        blk = _find_block(g, r, c, 2, 2)
        if blk is not None:
            r0, c0 = blk
            col = _top_color(g, c0)
            if col is not None:
                for rr in range(r0-1, r0+3):
                    for cc in range(c0-1, c0+3):
                        if 0 <= rr < H and 0 <= cc < W:
                            g[rr, cc] = col
            slot = _find_cleared_slot(g)
            if slot is not None:
                sr, sc = slot
                for rr in range(sr, sr+6):
                    for cc in range(sc, sc+6):
                        if 0 <= rr < H and 0 <= cc < W:
                            g[rr, cc] = 4
                for rr in range(sr+2, sr+4):
                    for cc in range(sc+2, sc+4):
                        g[rr, cc] = 2
            for rr in range(H-1, max(H-8, 0)-1, -1):
                cols = np.where(g[rr] == 3)[0]
                if len(cols) > 0:
                    cc = int(cols[0])
                    g[rr, cc] = 2
                    if cc-1 >= 0 and g[rr, cc-1] == 2:
                        g[rr, cc-1] = 3
                    break
            return g
    return g

def is_level_complete(grid):
    g = np.array(grid)
    bottom = g[57:61, :]
    return not (np.any(bottom == 9) or np.any(bottom == 14) or np.any(bottom == 11) or np.any(bottom == 15))
