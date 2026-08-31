import numpy as np

import numpy as np
from collections import deque


def _find_crate(g):
    """Locate the 'crate' object: a solid block of color 12 whose lower rows
    continue uniformly in color 9 (same column span). Returns (r0,r1,c0,c1)
    of the full crate bbox, or None."""
    H, W = g.shape
    mask = g == 12
    if not mask.any():
        return None
    seen = np.zeros((H, W), bool)
    best = None
    for i in range(H):
        for j in range(W):
            if mask[i, j] and not seen[i, j]:
                q = deque([(i, j)])
                seen[i, j] = True
                comp = []
                while q:
                    y, x = q.popleft()
                    comp.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
                r0 = min(y for y, _ in comp)
                r1 = max(y for y, _ in comp)
                c0 = min(x for _, x in comp)
                c1 = max(x for _, x in comp)
                if best is None or len(comp) > best[0]:
                    best = (len(comp), r0, r1, c0, c1)
    if best is None:
        return None
    _, r0, r1, c0, c1 = best
    rr = r1
    while rr + 1 < H and all(g[rr + 1, x] == 9 for x in range(c0, c1 + 1)):
        rr += 1
    return (r0, rr, c0, c1)


def _snap_crate_left(g):
    """Slide the crate left until its right edge aligns with the right edge of
    the open floor pocket (color-4 run) directly above it. Old cells refill
    with track color 3."""
    cr = _find_crate(g)
    if cr is None:
        return g
    r0, r1, c0, c1 = cr
    w = c1 - c0 + 1
    H, W = g.shape
    if r0 - 1 < 0:
        return g
    row = g[r0 - 1]
    target_right = None
    j = 0
    while j < W:
        if row[j] == 4:
            k = j
            while k + 1 < W and row[k + 1] == 4:
                k += 1
            # run overlaps the crate horizontally or sits immediately left of it
            if (j <= c1 and k >= c0) or (k + 1 == c0):
                target_right = k
                break
            j = k + 1
        else:
            j += 1
    if target_right is None:
        return g
    nc0 = target_right - w + 1
    if nc0 < 0 or nc0 + w > W or nc0 == c0:
        return g
    # capture pattern, verify destination is walkable track/floor
    pat = [g[y, c0:c1 + 1].copy() for y in range(r0, r1 + 1)]
    for y in range(r0, r1 + 1):
        for x in range(nc0, nc0 + w):
            if g[y, x] not in (3, 4):
                return g
    out = g.copy()
    for y in range(r0, r1 + 1):
        out[y, c0:c1 + 1] = 3
    for i, y in enumerate(range(r0, r1 + 1)):
        out[y, nc0:nc0 + w] = pat[i]
    return out


def _bar_tick(g):
    """The long color-11 bar advances one column per action tick: its leftmost
    column converts to track color 3."""
    idx = np.argwhere(g == 11)
    if len(idx) == 0:
        return g
    cmin = int(idx[:, 1].min())
    rows = idx[idx[:, 1] == cmin][:, 0]
    out = g.copy()
    out[rows, cmin] = 3
    return out


def engine(grid, action, data):
    g = np.asarray(grid).astype(int).copy()
    try:
        a = int(action)
    except Exception:
        return g
    if a == 6:  # click: no observed effect
        return g
    if a == 3:
        g = _snap_crate_left(g)
    # every keyboard/directional press advances the bar by one column
    if a in (1, 2, 3, 4, 5, 7):
        g = _bar_tick(g)
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    return bool(not np.any(g == 11))

import numpy as np

def is_level_complete(grid):
    g = np.asarray(grid)
    if g.size == 0:
        return False
    return bool((g == 0).all())
