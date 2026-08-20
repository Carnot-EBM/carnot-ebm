import numpy as np

def _cc(grid, r, c, val):
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W) or grid[r, c] != val:
        return []
    seen = {(r, c)}
    stack = [(r, c)]
    while stack:
        rr, cc = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and grid[nr, nc] == val:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return sorted(seen)

def _diamond(center, radius):
    r0, c0 = center
    out = []
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) <= radius:
                out.append((r0 + dr, c0 + dc))
    return out

def engine(grid, action, data):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    if action == 6 and data is not None:
        x, y = int(data.get('x', 0)), int(data.get('y', 0))
        c, r = x, y
        if 0 <= r < H and 0 <= c < W:
            v = g[r, c]
            if v == 15:
                # erase diamond radius 2 around click (center stays)
                for rr, cc in _diamond((r, c), 2):
                    if (rr, cc) != (r, c) and 0 <= rr < H and 0 <= cc < W:
                        g[rr, cc] = 0
            elif v == 3:
                # erase 4-connected 3 cluster containing click
                for rr, cc in _cc(g, r, c, 3):
                    g[rr, cc] = 0
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    return bool((g == 15).sum() == 0)