import numpy as np

def _blocks(g):
    mask = (g == 10)
    H, W = g.shape
    seen = np.zeros((H, W), bool)
    out = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and not seen[r, c]:
                stack = [(r, c)]
                seen[r, c] = True
                cells = []
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                        ny, nx = y+dy, x+dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                ys = [p[0] for p in cells]
                xs = [p[1] for p in cells]
                out.append((min(ys), min(xs), max(ys), max(xs), len(cells)))
    return out

def _move(g, dr, dc):
    g = g.copy()
    for (r0, c0, r1, c1, n) in _blocks(g):
        nr0, nc0 = r0+dr, c0+dc
        nr1, nc1 = r1+dr, c1+dc
        if nr0 < 0 or nc0 < 0 or nr1 >= g.shape[0] or nc1 >= g.shape[1]:
            continue
        if not np.all(g[nr0:nr1+1, nc0:nc1+1] == 5):
            continue
        g[r0:r1+1, c0:c1+1] = 5
        g[nr0:nr1+1, nc0:nc1+1] = 10
    return g

def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)
    if action == 1:
        return _move(g, -1, 0)
    if action == 2:
        return _move(g, 1, 0)
    if action == 3:
        return _move(g, 0, -1)
    if action == 4:
        return _move(g, 0, 1)
    return g

def is_level_complete(grid):
    return False
