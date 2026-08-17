import numpy as np

TARGET_COLORS = (8, 14, 9)

# Bottom template positions for this level layout.
BOTTOM_POS = {
    8: (57, 27),
    14: (57, 33),
    9: (57, 39),
}

# Static vertical track revealed when the head leaves it.
TRACK = (
    2, 2,
    3, 3, 3, 3,
    2, 2,
    3, 3, 3, 3,
    2, 2,
    3, 3, 3, 3,
    2, 2,
    3, 3, 3, 3,
    2, 2,
)

SEP_CAP = 5


def _body_color(r, c):
    return 2 if (r + c) % 3 == 1 else 1


def _find_head(g):
    H, W = g.shape
    max_r = min(H - 5, 48)  # keep top room above row 53
    for r in range(max_r):
        row = g[r]
        c = 0
        while c <= W - 6:
            if row[c] == 6 and row[c + 5] == 6:
                if np.all(row[c:c + 6] == 6) and np.all(g[r + 5, c:c + 6] == 6):
                    return (r, c)
            c += 1
    return None


def _find_top_targets(g):
    res = {}
    H, W = g.shape
    max_r = min(H - 3, 53)
    for C in TARGET_COLORS:
        found = False
        for r in range(max_r):
            sub = g[r:r + 4]
            for c in range(W - 3):
                if np.all(sub[:, c:c + 4] == C):
                    res[C] = (r, c)
                    found = True
                    break
            if found:
                break
    return res


def _scan_bottom_targets(g):
    res = {}
    H, W = g.shape
    start_r = min(53, H - 3)
    for C in TARGET_COLORS:
        best = None
        best_cnt = -1
        for r in range(start_r, H - 3):
            sub = g[r:r + 4]
            for c in range(W - 3):
                cnt = int(np.sum(sub[:, c:c + 4] == C))
                if cnt > best_cnt and cnt >= 8:
                    best_cnt = cnt
                    best = (r, c, cnt < 16)
        if best is not None:
            res[C] = best
    return res


def _get_bottom_info(g):
    H, W = g.shape
    info = {}
    for C, (ty, tx) in BOTTOM_POS.items():
        if ty + 4 <= H and tx + 4 <= W:
            cnt = int(np.sum(g[ty:ty + 4, tx:tx + 4] == C))
            if cnt >= 8:
                info[C] = (ty, tx, cnt < 16)

    if len(info) < len(TARGET_COLORS):
        fb = _scan_bottom_targets(g)
        for C, val in fb.items():
            if C not in info:
                info[C] = val
    return info


def _order_from_info(info):
    items = []
    for C in TARGET_COLORS:
        if C in info:
            items.append((info[C][1], C))
    items.sort()
    order = [C for _, C in items]
    if not order:
        order = list(TARGET_COLORS)
    return order


def _get_S(g):
    H, W = g.shape
    if H <= 53 or W <= 0:
        return 0
    s = 0
    c = W - 1
    while c >= 0 and g[53, c] == 3:
        s += 1
        c -= 1
    return s


def _increment_sep(g, S_old):
    H, W = g.shape
    if H > 53 and S_old < SEP_CAP:
        col = W - 1 - S_old
        if 0 <= col < W:
            g[53, col] = 3


def _base_top(r, c, x_head=None):
    if x_head is None:
        tc1, tc2 = 13, 14
    else:
        tc1, tc2 = x_head + 2, x_head + 3

    if c == tc1 or c == tc2:
        idx = r - 14
        if 0 <= idx < len(TRACK):
            return int(TRACK[idx])

    # Central top-room rectangle.
    if 12 <= r <= 41 and 17 <= c <= 46:
        return 4

    return 5


def _set_base_rect(g, r, c, h, w, x_head=None):
    H, W = g.shape
    for rr in range(max(0, r), min(H, r + h)):
        for cc in range(max(0, c), min(W, c + w)):
            g[rr, cc] = _base_top(rr, cc, x_head)


def _paint_body(g, y, x, E):
    H, W = g.shape
    x0 = x + 6
    if E < x0 or y + 3 >= H:
        return
    for c in range(x0, min(E, W - 1) + 1):
        g[y + 2, c] = _body_color(y + 2, c)
        g[y + 3, c] = _body_color(y + 3, c)


def _draw_target(g, C, ty, tx):
    H, W = g.shape
    if 0 <= ty < H - 3 and 0 <= tx < W - 3:
        g[ty:ty + 4, tx:tx + 4] = C


def _infer_E(g, y, x):
    H, W = g.shape
    x0 = x + 6
    if x0 >= W or y + 3 >= H:
        return x0 - 1

    tops = _find_top_targets(g)
    aligned = []
    for C, (ty, tx) in tops.items():
        if ty == y + 1:
            aligned.append((tx, tx + 3))

    E = x0 - 1
    for c in range(x0, W):
        inside_target = False
        for a, b in aligned:
            if a <= c <= b:
                inside_target = True
                break

        if not inside_target:
            if g[y + 2, c] == _body_color(y + 2, c) and g[y + 3, c] == _body_color(y + 3, c):
                E = c
            else:
                break
        else:
            E = c
    return E


def _dynamic_limit(g, y, x):
    H, W = g.shape
    x0 = x + 6
    if x0 >= W or y + 3 >= H:
        return x0 - 1

    r1, r2 = y + 2, y + 3
    limit = x0 - 1
    c = x0
    while c < W:
        if g[r1, c] == 5 and g[r2, c] == 5:
            break
        limit = c
        c += 1
    return max(limit, x0 - 1)


def _mark_bottom_hole(g, C):
    H, W = g.shape
    pos = BOTTOM_POS.get(C)
    if pos is None:
        fb = _scan_bottom_targets(g).get(C)
        if fb is None:
            return
        ty, tx = fb[0], fb[1]
    else:
        ty, tx = pos

    if ty + 4 > H or tx + 4 > W:
        return

    cnt = int(np.sum(g[ty:ty + 4, tx:tx + 4] == C))
    if cnt >= 16:
        g[ty + 1, tx + 1] = 0
        g[ty + 2, tx + 1] = 0


def _make_win_grid(shape):
    g = np.zeros(shape, dtype=int)
    if shape[0] >= 3 and shape[1] >= 3:
        g[0, 0] = 14
        g[0, 1] = 9
        g[0, 2] = 8
        g[1, 0] = 8
        g[1, 1] = 9
        g[1, 2] = 14
    return g


def _is_win_marker(g):
    H, W = g.shape
    if H < 3 or W < 3:
        return False
    return (
        g[0, 0] == 14 and g[0, 1] == 9 and g[0, 2] == 8 and
        g[1, 0] == 8 and g[1, 1] == 9 and g[1, 2] == 14
    )


def _vertical_move(g, y, x, dy):
    H, W = g.shape
    ny = y + dy

    top_limit = min(53, H)
    if ny < 0 or ny + 5 >= top_limit:
        return g

    S_old = _get_S(g)
    patch = g[y:y + 6, x:x + 6].copy()
    old_E = _infer_E(g, y, x)

    bottom_info = _get_bottom_info(g)
    collected = {C for C, (_, _, hole) in bottom_info.items() if hole}
    active = len(collected) > 0

    tops = _find_top_targets(g)
    new_pos = {}
    for C, (ty, tx) in tops.items():
        if C in collected and ty == y + 1:
            new_pos[C] = (ny + 1, tx)
        else:
            new_pos[C] = (ty, tx)

    # Head always leaves its old footprint.
    _set_base_rect(g, y, x, 6, 6, x)

    if active:
        x0 = x + 6
        if old_E >= x0:
            for rr in (y + 2, y + 3):
                for cc in range(x0, min(old_E, W - 1) + 1):
                    g[rr, cc] = _base_top(rr, cc, x)

        for C, (ty, tx) in tops.items():
            if C in collected and ty == y + 1:
                _set_base_rect(g, ty, tx, 4, 4, x)

    g[ny:ny + 6, x:x + 6] = patch
    _paint_body(g, ny, x, old_E)

    for C, (nty, ntx) in new_pos.items():
        _draw_target(g, C, nty, ntx)

    order = _order_from_info(bottom_info)
    nextC = None
    for C in order:
        if C not in collected:
            nextC = C
            break

    if len(collected) > 0 and nextC is not None:
        tpos = tops.get(nextC)
        if tpos is not None and tpos[0] == ny + 1:
            _increment_sep(g, S_old)

    return g


def _shift_targets(g, y, x):
    H, W = g.shape
    bottom_info = _get_bottom_info(g)
    collected = {C for C, (_, _, hole) in bottom_info.items() if hole}
    tops = _find_top_targets(g)

    attached = {}
    for C, (ty, tx) in tops.items():
        if C in collected and ty == y + 1:
            attached[C] = (ty, tx)

    if not attached:
        return g

    old_E = _infer_E(g, y, x)
    x0 = x + 6

    if old_E >= x0:
        for rr in (y + 2, y + 3):
            for cc in range(x0, min(old_E, W - 1) + 1):
                g[rr, cc] = _base_top(rr, cc, x)

    for C, (ty, tx) in attached.items():
        _set_base_rect(g, ty, tx, 4, 4, x)

    new_pos = {}
    maxE = x0 - 1
    for C, (ty, tx) in attached.items():
        nx = tx - 6
        if nx < 0:
            nx = 0
        new_pos[C] = (y + 1, nx)
        maxE = max(maxE, nx + 4)

    for C, (ty, tx) in tops.items():
        if C not in attached:
            new_pos[C] = (ty, tx)

    E_new = max(maxE, x0 - 1)
    _paint_body(g, y, x, E_new)

    for C, (nty, ntx) in new_pos.items():
        _draw_target(g, C, nty, ntx)

    return g


def _extend_body(g, y, x):
    H, W = g.shape
    S_old = _get_S(g)

    bottom_info = _get_bottom_info(g)
    collected = {C for C, (_, _, hole) in bottom_info.items() if hole}
    tops = _find_top_targets(g)

    old_E = _infer_E(g, y, x)
    limit = _dynamic_limit(g, y, x)
    x0 = x + 6

    if old_E < x0 - 1 or old_E >= limit:
        return g

    new_E = min(old_E + 6, limit)

    aligned = []
    reached = []
    for C, (ty, tx) in tops.items():
        if ty == y + 1:
            a, b = tx, tx + 3
            aligned.append((C, a, b))
            if a <= new_E and b >= old_E + 1 and C not in collected:
                reached.append(C)

    for c in range(old_E + 1, new_E + 1):
        inside_target = False
        for C, a, b in aligned:
            if a <= c <= b:
                inside_target = True
                break
        if not inside_target:
            g[y + 2, c] = _body_color(y + 2, c)
            g[y + 3, c] = _body_color(y + 3, c)

    # Restore target blocks over any body cells that may have touched them.
    for C, (ty, tx) in tops.items():
        _draw_target(g, C, ty, tx)

    order = _order_from_info(bottom_info)
    inc = False

    if reached:
        for C in reached:
            _mark_bottom_hole(g, C)

        if order and order[0] in reached:
            inc = True

        info2 = _get_bottom_info(g)
        holes = {C for C, (_, _, hole) in info2.items() if hole}
        if all(C in holes for C in TARGET_COLORS):
            return _make_win_grid(g.shape)
    else:
        if S_old == 0 and len(collected) == 0 and old_E <= x0 + 5:
            inc = True

    if inc:
        _increment_sep(g, S_old)

    return g


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)

    if _is_win_marker(g):
        return g

    head = _find_head(g)
    if head is None:
        return g

    y, x = head

    if action == 1:
        return _vertical_move(g, y, x, -6)
    if action == 2:
        return _vertical_move(g, y, x, 6)
    if action == 3:
        return _shift_targets(g, y, x)
    if action == 4:
        return _extend_body(g, y, x)

    # Unobserved actions are treated as no-ops.
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    if g.ndim != 2:
        return False

    H, W = g.shape

    if _is_win_marker(g):
        return True

    info = _get_bottom_info(g)
    holes = {C for C, (_, _, hole) in info.items() if hole}

    if not all(C in holes for C in TARGET_COLORS):
        return False

    head = _find_head(g)
    if head is None:
        return False

    y, x = head
    tops = _find_top_targets(g)

    if len(tops) != len(TARGET_COLORS):
        return False

    if not all(ty == y + 1 for ty, tx in tops.values()):
        return False

    E = _infer_E(g, y, x)
    max_right = max(tx + 3 for _, (ty, tx) in tops.items())

    return E >= max_right