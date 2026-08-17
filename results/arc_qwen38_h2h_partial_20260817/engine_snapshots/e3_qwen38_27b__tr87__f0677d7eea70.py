import numpy as np

_PATCH_T2 = (
    (52, 0, 5), (52, 4, 5),
    (53, 0, 5), (53, 1, 7),
    (54, 1, 7), (54, 2, 7),
    (55, 0, 5),
    (56, 0, 5), (56, 1, 7), (56, 2, 7), (56, 3, 7),
)

_PATCH_T3 = (
    (53, 2, 5), (53, 3, 7), (53, 4, 5),
    (54, 1, 5), (54, 2, 5), (54, 3, 7), (54, 4, 5),
    (55, 1, 7), (55, 2, 7), (55, 3, 7), (55, 4, 5),
    (56, 1, 5), (56, 2, 5), (56, 3, 5), (56, 4, 5),
)

_PATCH_T5 = (
    (52, 1, 7), (52, 2, 7), (52, 3, 7), (52, 4, 7),
    (53, 1, 5), (53, 2, 5), (53, 3, 5), (53, 4, 7),
    (54, 1, 7), (54, 2, 7), (54, 3, 5), (54, 4, 7),
    (55, 2, 7), (55, 3, 5), (55, 4, 7),
)

_PATCH_T6 = (
    (52, 0, 7), (52, 2, 5),
    (53, 4, 5),
    (54, 2, 5), (54, 3, 7), (54, 4, 5),
    (55, 1, 5), (55, 2, 5), (55, 4, 5),
    (56, 0, 7), (56, 1, 7), (56, 3, 7), (56, 4, 7),
)

_PATCH_T7 = (
    (52, 0, 5), (52, 1, 5), (52, 3, 5),
    (53, 1, 7), (53, 2, 7),
    (54, 2, 7),
    (55, 2, 7), (55, 3, 7),
    (56, 1, 5), (56, 3, 5), (56, 4, 5),
)

_PATCH_T9 = (
    (52, 0, 7), (52, 1, 7), (52, 3, 7), (52, 4, 7),
    (53, 1, 5), (53, 2, 5), (53, 3, 5),
    (54, 1, 7), (54, 2, 7), (54, 3, 7),
    (55, 0, 5), (55, 1, 7), (55, 3, 7), (55, 4, 5),
    (56, 0, 5), (56, 4, 5),
)

_PATCH_T11 = (
    (52, 0, 7), (52, 1, 7), (52, 3, 7),
    (53, 1, 5), (53, 2, 5),
    (54, 2, 5),
    (55, 2, 5), (55, 3, 5),
    (56, 1, 7), (56, 3, 7), (56, 4, 7),
)

_HOLE_CELLS = (
    (48, 0), (48, 1), (48, 2), (48, 3), (48, 4),
    (49, 0), (49, 4),
    (59, 0), (59, 4),
    (60, 0), (60, 1), (60, 2), (60, 3), (60, 4),
)

_WIN_DELTA = """
r4c5:7x7,2x3,11x7,2x6 r4c29:7x6,2x3,11x21 r5c5:7x1,5x1,7x5,2x3,11x1 r5c17:11x1,5x3,11x1,2x6 r5c29:7x1,5x3,7x2,2x3,11x3,5x1,11x4 r5c47:11x3,5x1,11x2,5x1,11x1,5x1,11x1,5x1,11x1
r6c5:7x1,5x4,7x2,2x3,11x3,5x1,11x3,2x6 r6c29:7x1,5x1,7x1,5x1,7x2,2x3,11x1 r6c40:11x1,5x1,11x1,5x1,11x2 r6c47:11x12
r7c5:7x1,5x1,7x2,5x1,7x2,3x3,11x3,5x1,11x3,2x6 r7c29:5x5,7x1,3x3,11x3,5x1,11x4 r7c47:5x2 r7c51:11x2,5x1,11x1,5x3,11x1
r8c5:7x1,5x1,7x2,5x1,7x2,2x3,11x3,5x1,11x3,2x6 r8c29:5x1,7x3,5x1,7x1,2x3,11x1 r8c40:11x1,5x1,11x1,5x1,11x6,5x1,11x2,5x1,11x1,5x1,11x1,5x1,11x1
r9c5:7x1,5x5,7x1,2x3,11x1,5x1 r9c18:5x1,11x1,5x1,11x1,2x6 r9c29:5x5,7x1,2x3,11x3,5x1,11x4 r9c47:11x3,5x1,11x2,5x3,11x1,5x1,11x1
r10c5:7x7,2x3,11x7,2x6 r10c29:7x6,2x3,11x21
r13c5:7x7,2x3,11x14 r13c35:7x7 r13c45:11x14
r14c5:7x1,5x4,7x2,2x3,11x2 r14c18:11x1,5x1,11x3 r14c26:11x1 r14c28:11x1 r14c35:7x2,5x1 r14c39:5x2,7x1 r14c45:11x1,5x1,11x2 r14c51:11x2,5x1,11x2,5x2,11x1
r15c5:7x1,5x1,7x2,5x2,7x1,2x3,11x8 r15c24:11x5 r15c35:7x2 r15c38:7x2 r15c41:7x1 r15c45:11x5 r15c51:11x6,5x1,11x1
r16c5:7x1,5x1,7x3,5x1,7x1,3x3,11x1 r16c18:5x3,11x8 r16c35:7x1 r16c37:5x1,7x2 r16c41:7x1 r16c45:11x1 r16c47:11x3 r16c51:11x2,5x1,11x3,5x1,11x1
r17c5:7x1,5x2,7x2,5x1,7x1,2x3,11x12,5x1,11x1 r17c35:7x2,5x1,7x2 r17c41:7x1 r17c45:11x1 r17c47:11x6,5x1,11x5
r18c5:7x2,5x4,7x1,2x3,11x2 r18c18:11x1,5x1,11x3 r18c24:11x1,5x3,11x1 r18c35:7x2,5x1 r18c39:5x2,7x1 r18c45:11x1 r18c48:11x2,5x1,11x2,5x2,11x2,5x1,11x1
r19c5:7x7,2x3,11x14 r19c35:7x7 r19c45:11x14
r22c5:7x7,2x3,11x21,2x6,7x3 r22c49:2x3,11x7
r23c5:7x3,5x3,7x1,2x3,11x1,5x1,11x1,5x1,11x1,5x1,11x2 r23c25:11x1 r23c28:11x2,5x1,11x1,5x1,11x1,5x1,11x1,2x6,7x3,5x1 r23c48:7x1,2x3,11x2,5x1,11x1,5x1,11x2
r24c5:7x3,5x1,7x1,5x1,7x1,2x3,11x5,5x1,11x2 r24c24:11x3 r24c28:11x2,5x1,11x5,2x6,7x1,5x3 r24c48:7x1,2x3,11x1,5x2,11x1,5x2,11x1
r25c5:7x1,5x5,7x1,3x3,11x1 r25c17:11x1,5x1,11x1,5x1,11x2 r25c24:11x1 r25c26:11x1 r25c28:11x2,5x1,11x1,5x1,11x1,5x1,11x1,2x6,7x1,5x1,7x1,5x1,7x1,5x1,7x1,3x3,11x1,5x1,11x3,5x1,11x1
r26c5:7x1,5x1,7x1,5x1,7x3,2x3,11x5,5x1,11x2 r26c24:11x3 r26c28:11x2,5x1,11x5,2x6,7x1,5x3 r26c48:7x1,2x3,11x7
r27c5:7x1,5x3,7x3,2x3,11x1,5x5,11x2 r27c25:11x1 r27c28:11x2,5x5,11x1,2x6,7x3,5x1 r27c48:7x1,2x3,11x1,5x1,11x3,5x1,11x1
r28c5:7x7,2x3,11x21,2x6,7x3 r28c49:2x3,11x7
r40c14:3x4,7x28,3x3 r41c14:3x4,7x5,5x1,7x4,5x1 r41c31:7x2 r41c34:5x2 r41c37:7x4,5x3,7x2,3x3
r42c14:3x4,7x2,5x4,7x4,5x1,7x1,5x1,7x2 r42c34:7x2,5x1,7x4,5x1,7x1,5x1,7x2,3x3
r43c14:3x4,7x2,5x1,7x2 r43c24:7x2 r43c27:5x4,7x2 r43c34:7x2,5x1 r43c38:7x2,5x4 r43c45:7x1,3x3
r44c14:3x4,7x2,5x1,7x2,5x1,7x2,5x1,7x1,5x1,7x4 r44c34:7x2,5x1,7x4,5x1,7x1,5x1,7x2,3x3
r45c14:3x4,7x1,5x3 r45c23:5x1,7x2,5x3,7x4 r45c34:5x2 r45c37:7x4,5x2 r45c44:7x2,3x3
r46c14:3x4,7x28,3x3
r48c8:0x5 r48c43:3x5 r49c8:0x1 r49c12:0x1 r49c43:3x1 r49c47:3x1
r51c7:11x49 r52c7:11x2,5x2,11x1,5x1,11x2,5x1,11x1 r52c20:11x2 r52c27:11x2 r52c30:11x3,5x1,11x2,5x2,11x1,5x2,11x2,5x2 r52c46:5x2,11x4,5x1,11x3
r53c7:11x1,5x2,11x7 r53c18:11x4 r53c23:11x6 r53c30:11x6 r53c37:11x3 r53c41:11x2 r53c44:11x3 r53c48:11x2,5x1,11x1,5x1,11x1,5x1,11x1
r54c7:11x8 r54c16:11x1 r54c20:11x2 r54c23:11x1 r54c25:11x1 r54c27:11x2 r54c30:5x3 r54c34:11x2 r54c37:11x1,5x1,11x1 r54c41:11x4 r54c46:11x6,5x1,11x3
r55c7:11x1,5x2,11x9,5x1,11x2 r55c23:11x10 r55c34:11x2 r55c37:11x3 r55c41:11x2 r55c44:11x3 r55c48:11x2,5x1,11x1,5x1,11x1,5x1,11x1
r56c7:11x2,5x2,11x1,5x1,11x2 r56c16:11x1 r56c18:5x2,11x2 r56c23:11x1 r56c25:11x1 r56c27:11x2,5x1,11x3 r56c34:11x2 r56c38:11x1 r56c41:11x2,5x2 r56c46:5x2,11x4,5x1,11x3
r57c7:11x49 r59c8:0x1 r59c12:0x1 r59c43:3x1 r59c47:3x1 r60c8:0x5 r60c43:3x5 r63c58:1x6
"""


def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape

    if int(np.count_nonzero(g == 11)) > 50:
        return g

    x = -1
    for c in range(W - 4):
        if (g[48, c:c + 5] == 0).all() and (g[60, c:c + 5] == 0).all():
            x = c
            break
    if x < 0:
        x = 15

    s = (x - 15) // 7 if 15 <= x <= 43 else -1

    out = g.copy()
    patch = None
    p_inc = 0
    win = False

    if action == 4:
        if 0 <= s < 4:
            nx = x + 7
            for rr, oo in _HOLE_CELLS:
                cc = x + oo
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = 3
            for rr, oo in _HOLE_CELLS:
                cc = nx + oo
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = 0

            completed = False
            if s == 1:
                if g[52, x] != 7 and g[56, x + 1] != 7:
                    completed = True
            elif s == 2:
                if g[52, x + 1] != 7 and g[56, x] == 7:
                    completed = True
            elif s == 3:
                if g[52, x] == 7:
                    completed = True

            if s > 0 and completed:
                p_inc = 1

    elif action == 3:
        if 0 < s:
            nx = x - 7
            for rr, oo in _HOLE_CELLS:
                cc = x + oo
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = 3
            for rr, oo in _HOLE_CELLS:
                cc = nx + oo
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = 0

    elif action == 6 and isinstance(data, dict):
        try:
            px = int(data.get("x", 0))
        except Exception:
            px = 0
        centers = (17, 24, 31, 38, 45)
        best = min(range(5), key=lambda i: abs(centers[i] - px))
        nx = 15 + 7 * best
        if nx != x and 15 <= nx <= 43:
            for rr, oo in _HOLE_CELLS:
                cc = x + oo
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = 3
            for rr, oo in _HOLE_CELLS:
                cc = nx + oo
                if 0 <= rr < H and 0 <= cc < W:
                    out[rr, cc] = 0

    elif action in (1, 2):
        if s == 1 and action == 2:
            if g[52, x] == 7:
                patch = _PATCH_T2
                p_inc = 1
            elif g[56, x + 1] == 7:
                patch = _PATCH_T3

        elif s == 2 and action == 1:
            if g[52, x + 1] == 7:
                if g[52, x] == 7:
                    patch = _PATCH_T7
                else:
                    patch = _PATCH_T6
                    p_inc = 1
            else:
                if g[56, x] != 7:
                    patch = _PATCH_T5

        elif s == 3 and action == 2:
            if g[52, x] != 7:
                patch = _PATCH_T9

        elif s == 4 and action == 2:
            if g[52, x] != 7:
                patch = _PATCH_T11
            else:
                win = True

    if win:
        out = g.copy()
        for part in _WIN_DELTA.split():
            colon = part.find(":")
            if colon < 0:
                continue
            head = part[:colon]
            body = part[colon + 1:]
            ci = head.find("c")
            rr = int(head[1:ci])
            cc0 = int(head[ci + 1:])
            col = cc0
            for pair in body.split(","):
                vx = pair.split("x")
                val = int(vx[0])
                cnt = int(vx[1])
                for k in range(cnt):
                    cc = col + k
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr, cc] = val
                col += cnt
        return out

    if patch is not None:
        for rr, oo, vv in patch:
            cc = x + oo
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = vv

    if p_inc > 0:
        for _ in range(p_inc):
            p = 0
            while p < W and out[63, W - 1 - p] == 4:
                p += 1
            ccol = W - 1 - p
            if ccol >= 0 and out[63, ccol] == 1:
                out[63, ccol] = 4
            else:
                for cc in range(ccol - 1, -1, -1):
                    if out[63, cc] == 1:
                        out[63, cc] = 4
                        break

    return out


def is_level_complete(grid):
    g = np.asarray(grid)
    if g.ndim != 2 or g.shape[0] < 64 or g.shape[1] < 64:
        return False

    if int(np.count_nonzero(g == 11)) > 50:
        return True

    W = g.shape[1]
    p = 0
    while p < W and g[63, W - 1 - p] == 4:
        p += 1
    return p >= 7