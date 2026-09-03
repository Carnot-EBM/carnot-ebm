import numpy as np
from collections import deque


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)
    if g.size == 0:
        return g

    try:
        act = int(action)
    except Exception:
        return g

    # Only the observed click behavior is applied here:
    # clicking an interactive core/button lights the top-left status cell to 5.
    if act != 6:
        return g

    H, W = g.shape
    if H == 0 or W == 0:
        return g

    px = None
    py = None
    if isinstance(data, dict):
        px = data.get("x")
        py = data.get("y")

    if px is None or py is None:
        return g

    try:
        x = int(px)
        y = int(py)
    except Exception:
        return g

    if y < 0 or y >= H or x < 0 or x >= W:
        return g

    # The clicked pixel must be part of a small special object.
    if g[y, x] not in (6, 15, 12):
        return g

    # Find a "button/core": a color-6 pixel whose immediate non-center neighbors
    # are uniformly one container color (observed: 15 or 12).
    for by in range(H):
        for bx in range(W):
            if g[by, bx] != 6:
                continue

            vals = set()
            ok = True
            for dy in (-1, 0, 1):
                ny = by + dy
                if ny < 0 or ny >= H:
                    ok = False
                    break
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    nx = bx + dx
                    if nx < 0 or nx >= W:
                        ok = False
                        break
                    v = g[ny, nx]
                    if v != 6:
                        vals.add(v)

            if ok and len(vals) == 1 and (vals == {15} or vals == {12}):
                if abs(by - y) <= 2 and abs(bx - x) <= 2:
                    g[0, 0] = 5
                    return g

    return g


def is_level_complete(grid):
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    if H == 0 or W == 0:
        return False

    # A fully filled left-edge status column is treated as complete.
    if np.all(g[:, 0] == 5):
        return True

    meter = 0
    while meter < H and g[meter, 0] == 5:
        meter += 1

    # If the interactive core has disappeared after some progress, treat as win.
    if meter > 0 and not np.any(g == 6):
        return True

    # Specific path-completion heuristic for this level family:
    # a color-3 source marker must be connected by an 8-connected color-1 trail
    # to a color-15 target sitting in a zero pocket.
    ones = (g == 1)
    if not np.any(ones):
        return False

    goal_zone = np.zeros((H, W), dtype=bool)
    any_goal = False

    for y in range(H):
        row = g[y]
        for x in range(W):
            if row[x] != 15:
                continue

            z = 0
            if y > 0 and g[y - 1, x] == 0:
                z += 1
            if y + 1 < H and g[y + 1, x] == 0:
                z += 1
            if x > 0 and g[y, x - 1] == 0:
                z += 1
            if x + 1 < W and g[y, x + 1] == 0:
                z += 1

            if z >= 3:
                any_goal = True
                y0 = max(0, y - 2)
                y1 = min(H - 1, y + 2)
                x0 = max(0, x - 2)
                x1 = min(W - 1, x + 2)
                goal_zone[y0:y1 + 1, x0:x1 + 1] = True

    if not any_goal:
        return False

    src_touch = np.zeros((H, W), dtype=bool)
    any_src = False

    for y in range(H):
        for x in range(W):
            if g[y, x] != 3:
                continue

            found = False
            for dy in (-1, 0, 1):
                ny = y + dy
                if ny < 0 or ny >= H:
                    continue
                for dx in (-1, 0, 1):
                    nx = x + dx
                    if nx < 0 or nx >= W:
                        continue
                    if g[ny, nx] == 1:
                        src_touch[ny, nx] = True
                        found = True

            if found:
                any_src = True

    if not any_src:
        return False

    visited = np.zeros((H, W), dtype=bool)
    dirs8 = (
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    )

    for sy in range(H):
        for sx in range(W):
            if not ones[sy, sx] or visited[sy, sx]:
                continue

            q = deque([(sy, sx)])
            visited[sy, sx] = True
            touches_source = bool(src_touch[sy, sx])
            touches_goal = bool(goal_zone[sy, sx])

            while q:
                cy, cx = q.popleft()

                if touches_source and touches_goal:
                    return True

                for dy, dx in dirs8:
                    ny = cy + dy
                    nx = cx + dx
                    if 0 <= ny < H and 0 <= nx < W:
                        if ones[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            if src_touch[ny, nx]:
                                touches_source = True
                            if goal_zone[ny, nx]:
                                touches_goal = True
                            q.append((ny, nx))

    return False