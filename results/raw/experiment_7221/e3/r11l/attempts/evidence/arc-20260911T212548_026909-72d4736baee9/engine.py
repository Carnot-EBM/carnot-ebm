import numpy as np

def _diamond_cells(cy, cx):
    cells = []
    for dy in range(-2, 3):
        for dx in range(-2, 3):
            if abs(dy) + abs(dx) <= 2:
                cells.append((cy + dy, cx + dx))
    return cells

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    if action == 6 and data is not None:
        # 1) fill column 0 from the top with color 5 (one cell per click)
        r = 0
        while r < H and g[r, 0] == 0:
            r += 1
        if r < H:
            g[r, 0] = 5
        # 2) move the 6-centered diamond one step down-right
        ys, xs = np.where(g == 6)
        if len(ys) > 0:
            cy, cx = int(ys[0]), int(xs[0])
            ny, nx = cy + 1, cx + 1
            if 0 <= ny < H and 0 <= nx < W:
                old = set(_diamond_cells(cy, cx))
                new = set(_diamond_cells(ny, nx))
                for (rr, cc) in (new - old):
                    if 0 <= rr < H and 0 <= cc < W:
                        g[rr, cc] = 15
                for (rr, cc) in (old - new):
                    if 0 <= rr < H and 0 <= cc < W:
                        g[rr, cc] = 5
                g[ny, nx] = 6
    return g

def is_level_complete(grid):
    return False
