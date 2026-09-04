import numpy as np

_DELTA_STRS = (
    "r0c0:5x1 r34c7:5x1 r35c6:5x3 r36c5:5x5 r37c6:5x3 r38c7:5x1 r38c9:5x1 "
    "r39c10:5x1 r40c11:5x1 r41c12:5x1 r42c12:5x1 r43c13:5x1 r44c14:5x1 "
    "r45c15:5x2,0x1,5x1 r46c15:5x1,0x3,5x1 r47c15:0x2,15x1,0x2 "
    "r48c15:5x1,0x3,5x1 r49c16:5x1,0x1,5x1 r50c19:1x1,5x1 r51c21:15x3",

    "r1c0:5x1 r45c17:5x1 r46c16:5x3 r47c15:5x5 r48c16:5x3 r49c17:5x1 "
    "r49c19:5x1 r50c19:5x1 r51c20:5x2,0x1,5x1 r52c20:5x1,0x3,5x1 "
    "r53c20:0x2,15x1,0x2 r54c20:5x1,0x2 r54c25:15x1 r55c21:5x1 "
    "r55c24:15x3 r56c22:15x2,6x1,15x2 r57c22:15x5 r58c23:15x3",

    "r2c0:5x1 r51c22:5x1 r52c21:5x3 r53c20:5x5 r54c21:5x3,0x1,5x1 "
    "r55c22:5x1,0x1 r56c22:0x1 r56c24:15x1 r56c27:15x1 r57c22:5x1 "
    "r57c25:6x1 r57c27:15x1 r58c26:15x2 r59c24:15x3"
)

_DELTAS = []
for _s in _DELTA_STRS:
    _cells = []
    for _tok in _s.split():
        _left, _right = _tok.split(":")
        _ci = _left.find("c")
        _row = int(_left[1:_ci])
        _col = int(_left[_ci + 1:])
        for _pair in _right.split(","):
            _xi = _pair.find("x")
            _val = int(_pair[:_xi])
            _cnt = int(_pair[_xi + 1:])
            for _i in range(_cnt):
                _cc = _col + _i
                if _cc != 0:
                    _cells.append((_row, _cc, _val))
            _col += _cnt
    _DELTAS.append(tuple(_cells))

_THICK_OFFSETS = tuple(
    (dr, dc)
    for dr in (-2, -1, 0, 1, 2)
    for dc in (-2, -1, 0, 1, 2)
    if not (abs(dr) == 2 and abs(dc) == 2)
)

_ZERO_OFFSETS = (
    (-2, 0),
    (-1, -1), (-1, 0), (-1, 1),
    (0, -2), (0, -1), (0, 1), (0, 2),
    (1, -1), (1, 0), (1, 1),
    (2, 0),
)


def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    H, W = g.shape

    if H > 0 and W > 0:
        for r in range(H):
            if g[r, 0] != 5:
                g[r, 0] = 5
                break

    delta_idx = -1
    off_r = 0
    off_c = 0

    if action == 6 and data is not None:
        try:
            px = int(data.get("x", 0))
            py = int(data.get("y", 0))
        except Exception:
            px = 0
            py = 0

        ys, xs = np.where(g == 6)

        if ys.size == 1:
            gr = int(ys[0])
            gc = int(xs[0])

            full_body = True
            for dr, dc in _THICK_OFFSETS:
                rr = gr + dr
                cc = gc + dc
                if rr < 0 or rr >= H or cc < 0 or cc >= W or g[rr, cc] != 15:
                    full_body = False
                    break

            if full_body:
                back_path = False
                r0 = max(0, gr - 12)
                c0 = max(0, gc - 12)
                for rr in range(r0, gr):
                    row = g[rr]
                    for cc in range(c0, gc):
                        if row[cc] == 1:
                            back_path = True
                            break
                    if back_path:
                        break

                if back_path and py == gr and px == gc:
                    delta_idx = 0
                    off_r = gr - 47
                    off_c = gc - 17
            else:
                if py == gr and px == gc:
                    delta_idx = 2
                    off_r = gr - 56
                    off_c = gc - 24

        elif ys.size == 0:
            found_r = -1
            found_c = -1

            cand_ys, cand_xs = np.where(g == 15)
            for i in range(len(cand_ys)):
                cr = int(cand_ys[i])
                cc = int(cand_xs[i])
                ok = True
                for dr, dc in _ZERO_OFFSETS:
                    rr = cr + dr
                    ccol = cc + dc
                    if rr < 0 or rr >= H or ccol < 0 or ccol >= W or g[rr, ccol] != 0:
                        ok = False
                        break
                if ok:
                    found_r = cr
                    found_c = cc
                    break

            if found_r >= 0:
                ty, tx = np.where(g == 3)
                if ty.size > 0:
                    tr = int(np.sum(ty)) // int(ty.size)
                    tc = int(np.sum(tx)) // int(tx.size)
                    pr = (found_r + tr) // 2
                    pc = (found_c + tc) // 2
                else:
                    pr = found_r + 6
                    pc = found_c + 5

                if py == pr and px == pc:
                    delta_idx = 1
                    off_r = found_r - 47
                    off_c = found_c - 17

    if delta_idx >= 0:
        for r, c, v in _DELTAS[delta_idx]:
            nr = r + off_r
            nc = c + off_c
            if 0 <= nr < H and 0 <= nc < W:
                g[nr, nc] = v

    return g


def is_level_complete(grid):
    if grid.size == 0:
        return False

    g = np.asarray(grid)
    has_board = bool(np.any((g == 5) | (g == 2)))
    if not has_board:
        return False

    timer_full = bool(g.shape[1] > 0 and np.all(g[:, 0] == 5))
    target_gone = bool(not np.any(g == 3))

    return timer_full or target_gone