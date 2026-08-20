import numpy as np

def _largest_15(g):
    H, W = g.shape
    mask = (g == 15)
    best = None
    seen = np.zeros_like(mask, dtype=bool)
    from collections import deque
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                q = deque([(r, c)]); seen[r, c] = True; cells = []
                while q:
                    rr, cc = q.popleft(); cells.append((rr, cc))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True; q.append((nr, nc))
                if best is None or len(cells) > len(best):
                    best = cells
    return best

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    cells = _largest_15(g)
    if cells is None:
        return g
    rs = [r for r, c in cells]; cs = [c for r, c in cells]
    r0, r1 = min(rs), max(rs); c0, c1 = min(cs), max(cs)
    # rotation direction: action 2 = CCW, action 4 = CW (guess)
    if action == 2:
        rot = -1
    elif action == 4:
        rot = 1
    else:
        return g
    # rotation matrix
    if rot == 1:
        M = np.array([[0, -1], [1, 0]])
    else:
        M = np.array([[0, 1], [-1, 0]])
    # pivot: try center of block
    pr, pc = (r0 + r1) / 2.0, (c0 + c1) / 2.0
    newcells = set()
    for (r, c) in cells:
        v = np.array([r - pr, c - pc], dtype=float)
        nv = M @ v
        nr = int(round(pr + nv[0])); nc = int(round(pc + nv[1]))
        newcells.add((nr, nc))
    # clear old, set new
    for (r, c) in cells:
        g[r, c] = 5
    for (r, c) in newcells:
        if 0 <= r < H and 0 <= c < W:
            g[r, c] = 15
    return g

def is_level_complete(grid):
    return False