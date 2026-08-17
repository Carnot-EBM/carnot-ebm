# ruff: noqa
# INCIDENT FIXTURE (REQ-ARC-WMTE-6400) -- preserved VERBATIM below this header.
#
# This is the LLM-generated sb26 world-model engine that OOM-killed the agent
# process (~78 GB RSS, twice, same seed). The bug: `flood(r, c, val, visited=None)`
# is called WITHOUT `visited` at the two click branches, so any click on a >=2-cell
# same-valued component re-pushes cells forever and `comp` grows without bound.
#
# DO NOT FIX THE BUG. The bug IS the fixture: tests load `engine` from this file
# and prove arc_engine_call_guard converts the hang into a handled guard trip.
# Source: h2h harness output e3_w0_qwen38_27b/sb26/world_model.py (2026-08 run).
import numpy as np

_NONBLOCK = (0, 2, 3, 4, 5)
_BG = 4        # board background color
_MARK = 2      # empty-slot marker / consumed-block remnant / progress-line base
_BAR_FULL = 3  # filled progress cell


def engine(grid, action, data):
    g = grid.copy()
    H, W = g.shape

    def flood(r, c, val, visited=None):
        stack = [(r, c)]
        comp = []
        while stack:
            rr, cc = stack.pop()
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            if visited is not None and visited[rr, cc]:
                continue
            if int(g[rr, cc]) != val:
                continue
            if visited is not None:
                visited[rr, cc] = True
            comp.append((rr, cc))
            stack.extend(((rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)))
        return comp

    def bbox(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        return min(rs), max(rs), min(cs), max(cs)

    def progress_row():
        best_r, best_run = None, 0
        for r in range(H // 2, H):
            cur, run, mx = None, 0, 0
            for v in g[r]:
                v = int(v)
                if v == _MARK or v == _BAR_FULL:
                    if cur == v:
                        run += 1
                    else:
                        cur, run = v, 1
                    if run > mx:
                        mx = run
                else:
                    cur, run = None, 0
            if mx > best_run:
                best_run, best_r = mx, r
        if best_r is not None and best_run >= W // 2:
            return best_r
        return max(0, H - 11)

    def zero_ring(r0, r1, c0, c1):
        for rr in range(r0 - 1, r1 + 2):
            for cc in range(c0 - 1, c1 + 2):
                if rr != r0 and rr != r1 and cc != c0 and cc != c1:
                    continue
                if rr < 0 or rr >= H or cc < 0 or cc >= W or int(g[rr, cc]) != 0:
                    return False
        return True

    def selected_block():
        visited = np.zeros((H, W), bool)
        for r in range(H):
            for c in range(W):
                v = int(g[r, c])
                if v in _NONBLOCK or visited[r, c]:
                    continue
                comp = flood(r, c, v, visited)
                r0, r1, c0, c1 = bbox(comp)
                if len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1) and zero_ring(r0, r1, c0, c1):
                    return (r0, r1, c0, c1, v)
        return None

    def targets():
        out = []
        row = g[1]
        i, n = 0, row.shape[0]
        while i < n:
            v = int(row[i])
            j = i
            while j < n and int(row[j]) == v:
                j += 1
            if j - i >= 4 and v not in (4, 5):
                out.append(v)
            i = j
        return out

    def winning_config():
        t = targets()
        if not t:
            return False
        P = progress_row()
        sites = []
        visited = np.zeros((H, W), bool)
        for r in range(H):
            for c in range(W):
                v = int(g[r, c])
                if v in _NONBLOCK or visited[r, c]:
                    continue
                comp = flood(r, c, v, visited)
                r0, r1, c0, c1 = bbox(comp)
                if (len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1) and
                        (r1 - r0) == 3 and (c1 - c0) == 3 and r1 < P):
                    sites.append(((c0 + c1) // 2, v))
        visited = np.zeros((H, W), bool)
        for r in range(H):
            for c in range(W):
                if int(g[r, c]) != _MARK or visited[r, c]:
                    continue
                comp = flood(r, c, _MARK, visited)
                r0, r1, c0, c1 = bbox(comp)
                if len(comp) == 4 and (r1 - r0) == 1 and (c1 - c0) == 1 and r1 < P:
                    sites.append(((c0 + c1) // 2, None))
        sites.sort(key=lambda s: s[0])
        if len(sites) != len(t):
            return False
        vals = [s[1] for s in sites]
        if any(v is None for v in vals):
            return False
        return all(int(a) == int(b) for a, b in zip(vals, t))

    if action == 6 and isinstance(data, dict):
        try:
            col, row = int(data['x']), int(data['y'])
        except Exception:
            return g
        if 0 <= row < H and 0 <= col < W:
            P = progress_row()
            v = int(g[row, col])
            if v not in _NONBLOCK and row > P:
                # click on a bottom block -> select it (carve zero ring around it)
                comp = flood(row, col, v)
                r0, r1, c0, c1 = bbox(comp)
                if not zero_ring(r0, r1, c0, c1):
                    sel = selected_block()
                    if sel is not None:
                        sr0, sr1, sc0, sc1, _ = sel
                        for rr in range(sr0 - 1, sr1 + 2):
                            for cc in range(sc0 - 1, sc1 + 2):
                                if (rr == sr0 or rr == sr1 or cc == sc0 or cc == sc1) and int(g[rr, cc]) == 0:
                                    g[rr, cc] = _BG
                    for rr in range(r0 - 1, r1 + 2):
                        for cc in range(c0 - 1, c1 + 2):
                            if rr == r0 or rr == r1 or cc == c0 or cc == c1:
                                g[rr, cc] = 0
            elif v == _MARK and row < P:
                # click on an empty middle slot marker -> place selected block there
                comp = flood(row, col, v)
                if len(comp) == 4:
                    mr0, mr1, mc0, mc1 = bbox(comp)
                    if (mr1 - mr0) == 1 and (mc1 - mc0) == 1:
                        sel = selected_block()
                        if sel is not None:
                            r0, r1, c0, c1, cv = sel
                            # 4x4 colored block centered on the marker
                            for rr in range(mr0 - 1, mr1 + 2):
                                for cc in range(mc0 - 1, mc1 + 2):
                                    g[rr, cc] = cv
                            # bottom block consumed: whole area back to background...
                            for rr in range(r0 - 1, r1 + 2):
                                for cc in range(c0 - 1, c1 + 2):
                                    g[rr, cc] = _BG
                            # ...except a 2x2 remnant marker at its center
                            g[r0 + 1:r1, c0 + 1:c1] = _MARK
                            # progress line grows by one cell from the right edge
                            m = int(np.sum(g[P] == _BAR_FULL))
                            idx = W - 1 - m
                            if 0 <= idx < W:
                                g[P, idx] = _BAR_FULL
    elif action == 5:
        # submit: if every slot holds the target sequence, level is complete;
        # mark completion by filling the whole progress line.
        if winning_config():
            P = progress_row()
            g[P, :] = _BAR_FULL
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    H, W = g.shape
    if H < 2 or W == 0:
        return False

    def flood(r, c, val, visited=None):
        stack = [(r, c)]
        comp = []
        while stack:
            rr, cc = stack.pop()
            if rr < 0 or rr >= H or cc < 0 or cc >= W:
                continue
            if visited is not None and visited[rr, cc]:
                continue
            if int(g[rr, cc]) != val:
                continue
            if visited is not None:
                visited[rr, cc] = True
            comp.append((rr, cc))
            stack.extend(((rr + 1, cc), (rr - 1, cc), (rr, cc + 1), (rr, cc - 1)))
        return comp

    def bbox(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        return min(rs), max(rs), min(cs), max(cs)

    def progress_row():
        best_r, best_run = None, 0
        for r in range(H // 2, H):
            cur, run, mx = None, 0, 0
            for v in g[r]:
                v = int(v)
                if v == _MARK or v == _BAR_FULL:
                    if cur == v:
                        run += 1
                    else:
                        cur, run = v, 1
                    if run > mx:
                        mx = run
                else:
                    cur, run = None, 0
            if mx > best_run:
                best_run, best_r = mx, r
        if best_r is not None and best_run >= W // 2:
            return best_r
        return max(0, H - 11)

    def targets():
        out = []
        row = g[1]
        i, n = 0, row.shape[0]
        while i < n:
            v = int(row[i])
            j = i
            while j < n and int(row[j]) == v:
                j += 1
            if j - i >= 4 and v not in (4, 5):
                out.append(v)
            i = j
        return out

    t = targets()
    if not t:
        return False
    P = progress_row()
    # completion marker: the whole progress line filled
    if int(np.sum(g[P] == _BAR_FULL)) != W:
        return False
    sites = []
    visited = np.zeros((H, W), bool)
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v in _NONBLOCK or visited[r, c]:
                continue
            comp = flood(r, c, v, visited)
            r0, r1, c0, c1 = bbox(comp)
            if (len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1) and
                    (r1 - r0) == 3 and (c1 - c0) == 3 and r1 < P):
                sites.append(((c0 + c1) // 2, v))
    visited = np.zeros((H, W), bool)
    for r in range(H):
        for c in range(W):
            if int(g[r, c]) != _MARK or visited[r, c]:
                continue
            comp = flood(r, c, _MARK, visited)
            r0, r1, c0, c1 = bbox(comp)
            if len(comp) == 4 and (r1 - r0) == 1 and (c1 - c0) == 1 and r1 < P:
                sites.append(((c0 + c1) // 2, None))
    sites.sort(key=lambda s: s[0])
    if len(sites) != len(t):
        return False
    vals = [s[1] for s in sites]
    if any(v is None for v in vals):
        return False
    return all(int(a) == int(b) for a, b in zip(vals, t))