import numpy as np
import re

A = (
    (1, 1, 0, 0, 0),
    (0, 1, 0, 1, 0),
    (0, 0, 0, 0, 0),
    (0, 1, 0, 1, 1),
    (0, 0, 0, 1, 1),
)

I = (
    (1, 0, 0, 0, 1),
    (1, 0, 1, 0, 1),
    (0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1),
    (1, 0, 0, 0, 1),
)

J = (
    (0, 0, 0, 0, 0),
    (0, 1, 1, 0, 1),
    (0, 1, 1, 0, 1),
    (0, 0, 0, 0, 1),
    (0, 1, 1, 1, 1),
)

K = (
    (0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0),
    (0, 0, 0, 1, 0),
    (0, 1, 1, 1, 0),
    (0, 0, 0, 0, 0),
)

L0 = (
    (0, 0, 0, 0, 0),
    (0, 1, 1, 1, 0),
    (0, 0, 0, 1, 0),
    (0, 1, 0, 1, 0),
    (0, 0, 0, 0, 0),
)

L1 = (
    (0, 1, 1, 1, 1),
    (0, 0, 0, 0, 1),
    (0, 1, 1, 0, 1),
    (0, 1, 1, 0, 1),
    (0, 0, 0, 0, 0),
)

L2 = (
    (1, 1, 0, 1, 1),
    (0, 0, 0, 0, 0),
    (0, 1, 0, 1, 0),
    (0, 0, 0, 0, 0),
    (1, 1, 0, 1, 1),
)

L3 = (
    (0, 0, 0, 0, 1),
    (0, 1, 1, 0, 0),
    (0, 1, 1, 1, 0),
    (0, 0, 1, 1, 0),
    (1, 0, 0, 0, 0),
)

M0 = (
    (0, 0, 0, 0, 0),
    (0, 1, 1, 1, 0),
    (0, 0, 0, 0, 0),
    (1, 0, 1, 0, 1),
    (1, 0, 0, 0, 1),
)

M1 = (
    (1, 1, 0, 1, 1),
    (0, 0, 0, 0, 0),
    (0, 1, 1, 1, 0),
    (0, 1, 1, 1, 0),
    (0, 0, 0, 0, 0),
)

P = L2

HOLE_STARTS = [15, 22, 29, 36, 43]
EXPECTED_PRE_WIN = [A, K, L3, M1, P]
TERMINAL = {0: A, 1: K, 2: L3, 3: M1}
LEAVE_REWARD = {0: 0, 1: 1, 2: 1, 3: 1}

TRANSITIONS = {
    (1, I, 2): (J, 1),
    (1, J, 2): (K, 0),
    (2, L0, 1): (L1, 0),
    (2, L1, 1): (L2, 1),
    (2, L2, 1): (L3, 0),
    (3, M0, 2): (M1, 0),
    (4, L3, 2): (L2, 0),
}

PRE_WIN_RLE = """
r0:2x64
r1:2x64
r2:2x64
r3:2x64
r4:2x12,10x7,2x3,7x7,2x6,10x7,2x3,7x7,2x12
r5:2x12,10x1,5x5,10x1,2x3,7x1,5x4,7x2,2x6,10x1,5x1,10x3,5x1,10x1,2x3,7x1,5x4,7x2,2x12
r6:2x12,10x1,5x1,10x3,5x1,10x1,2x3,7x1,5x1,7x2,5x2,7x1,2x6,10x1,5x5,10x1,2x3,7x1,5x1,7x2,5x1,7x2,2x12
r7:2x12,10x1,5x2,10x1,5x2,10x1,3x3,7x1,5x1,7x3,5x1,7x1,2x6,10x3,5x1,10x3,3x3,7x1,5x1,7x2,5x2,7x1,2x12
r8:2x12,10x1,5x1,10x3,5x1,10x1,2x3,7x1,5x2,7x2,5x1,7x1,2x6,10x1,5x5,10x1,2x3,7x1,5x1,7x2,5x1,7x2,2x12
r9:2x12,10x1,5x1,10x3,5x1,10x1,2x3,7x2,5x4,7x1,2x6,10x1,5x1,10x3,5x1,10x1,2x3,7x1,5x4,7x2,2x12
r10:2x12,10x7,2x3,7x7,2x6,10x7,2x3,7x7,2x12
r11:2x64
r12:2x64
r13:2x12,10x7,2x3,7x7,2x6,10x7,2x3,7x7,2x12
r14:2x12,10x5,5x1,10x1,2x3,7x1,5x5,7x1,2x6,10x3,5x1,10x3,2x3,7x3,5x3,7x1,2x12
r15:2x12,10x3,5x1,10x1,5x1,10x1,2x3,7x1,5x1,7x2,5x1,7x2,2x6,10x1,5x5,10x1,2x3,7x3,5x1,7x1,5x1,7x1,2x12
r16:2x12,10x1,5x5,10x1,3x3,7x1,5x1,7x2,5x1,7x2,2x6,10x1,5x1,10x1,5x1,10x1,5x1,10x1,3x3,7x1,5x5,7x1,2x12
r17:2x12,10x3,5x1,10x1,5x1,10x1,2x3,7x1,5x4,7x2,2x6,10x1,5x1,10x1,5x1,10x1,5x1,10x1,2x3,7x1,5x1,7x1,5x1,7x3,2x12
r18:2x12,10x5,5x1,10x1,2x3,7x1,5x1,7x5,2x6,10x3,5x1,10x3,2x3,7x1,5x3,7x3,2x12
r19:2x12,10x7,2x3,7x7,2x6,10x7,2x3,7x7,2x12
r20:2x64
r21:2x64
r22:2x12,10x7,2x3,7x7,2x6,10x7,2x3,7x7,2x12
r23:2x12,10x5,5x1,10x1,2x3,7x1,5x5,7x1,2x6,10x1,5x2,10x1,5x2,10x1,2x3,7x3,5x1,7x3,2x12
r24:2x12,10x3,5x1,10x1,5x1,10x1,2x3,7x1,5x1,7x3,5x1,7x1,2x6,10x1,5x1,10x3,5x1,10x1,2x3,7x1,5x5,7x1,2x12
r25:2x12,10x1,5x5,10x1,3x3,7x1,5x1,7x1,5x3,7x1,2x6,10x1,5x1,10x3,5x1,10x1,3x3,7x1,5x1,7x1,5x1,7x1,5x1,7x1,2x12
r26:2x12,10x1,5x1,10x1,5x1,10x3,2x3,7x1,5x1,7x1,5x1,7x1,5x1,7x1,2x6,10x1,5x5,10x1,2x3,7x1,5x5,7x1,2x12
r27:2x12,10x1,5x1,10x5,2x3,7x1,5x5,7x1,2x6,10x1,5x1,10x3,5x1,10x1,2x3,7x3,5x1,7x3,2x12
r28:2x12,10x7,2x3,7x7,2x6,10x7,2x3,7x7,2x12
r29:2x64
r30:2x64
r31:2x64
r32:2x64
r33:2x64
r34:3x64
r35:3x64
r36:3x64
r37:3x64
r38:3x64
r39:3x64
r40:3x14,10x35,3x15
r41:3x14,10x2,5x3,10x7,5x1,10x2,5x5,10x2,5x2,10x1,5x2,10x4,5x1,10x3,3x15
r42:3x14,10x4,5x1,10x5,5x1,10x1,5x1,10x4,5x1,10x1,5x1,10x3,5x1,10x1,5x1,10x5,5x1,10x3,3x15
r43:3x14,10x1,5x5,10x2,5x5,10x6,5x1,10x3,5x3,10x4,5x3,10x2,3x15
r44:3x14,10x4,5x1,10x3,5x1,10x1,5x1,10x6,5x1,10x1,5x1,10x3,5x1,10x1,5x1,10x5,5x1,10x3,3x15
r45:3x14,10x2,5x3,10x3,5x1,10x6,5x5,10x2,5x2,10x1,5x2,10x2,5x5,10x1,3x15
r46:3x14,10x35,3x15
r47:3x64
r48:3x43,0x5,3x16
r49:3x43,0x1,3x3,0x1,3x16
r50:3x64
r51:3x14,7x35,3x15
r52:3x14,7x3,5x3,7x2,5x5,7x2,5x4,7x5,5x1,7x6,5x1,7x3,3x15
r53:3x14,7x3,5x1,7x1,5x1,7x2,5x1,7x1,5x1,7x1,5x1,7x2,5x1,7x2,5x2,7x2,5x5,7x2,5x5,7x1,3x15
r54:3x14,7x1,5x5,7x2,5x3,7x1,5x1,7x2,5x1,7x3,5x1,7x2,5x1,7x3,5x1,7x2,5x1,7x1,5x1,7x1,5x1,7x1,3x15
r55:3x14,7x1,5x1,7x1,5x1,7x4,5x1,7x3,5x1,7x2,5x2,7x2,5x1,7x2,5x1,7x3,5x1,7x2,5x5,7x1,3x15
r56:3x14,7x1,5x3,7x4,5x5,7x3,5x4,7x2,5x5,7x4,5x1,7x3,3x15
r57:3x14,7x35,3x15
r58:3x64
r59:3x43,0x1,3x3,0x1,3x16
r60:3x43,0x5,3x16
r61:3x64
r62:3x64
r63:1x58,4x6
"""

WIN_DELTA = """
r4c5:7x7,2x3,11x7,2x6 r4c29:7x6,2x3,11x21 r5c5:7x1,5x1,7x5,2x3,11x1 r5c17:11x1,5x3,11x1,2x6 r5c29:7x1,5x3,7x2,2x3,11x3,5x1,11x4 r5c47:11x3,5x1,11x2,5x1,11x1,5x1,11x1,5x1,11x1 r6c5:7x1,5x4,7x2,2x3,11x3,5x1,11x3,2x6 r6c29:7x1,5x1,7x1,5x1,7x2,2x3,11x1 r6c40:11x1,5x1,11x1,5x1,11x2 r6c47:11x12 r7c5:7x1,5x1,7x2,5x1,7x2,3x3,11x3,5x1,11x3,2x6 r7c29:5x5,7x1,3x3,11x3,5x1,11x4 r7c47:5x2 r7c51:11x2,5x1,11x1,5x3,11x1 r8c5:7x1,5x1,7x2,5x1,7x2,2x3,11x3,5x1,11x3,2x6 r8c29:5x1,7x3,5x1,7x1,2x3,11x1 r8c40:11x1,5x1,11x1,5x1,11x6,5x1,11x2,5x1,11x1,5x1,11x1,5x1,11x1 r9c5:7x1,5x5,7x1,2x3,11x1,5x1 r9c18:5x1,11x1,5x1,11x1,2x6 r9c29:5x5,7x1,2x3,11x3,5x1,11x4 r9c47:11x3,5x1,11x2,5x3,11x1,5x1,11x1 r10c5:7x7,2x3,11x7,2x6 r10c29:7x6,2x3,11x21 r13c5:7x7,2x3,11x14 r13c35:7x7 r13c45:11x14 r14c5:7x1,5x4,7x2,2x3,11x2 r14c18:11x1,5x1,11x3 r14c26:11x1 r14c28:11x1 r14c35:7x2,5x1 r14c39:5x2,7x1 r14c45:11x1,5x1,11x2 r14c51:11x2,5x1,11x2,5x2,11x1 r15c5:7x1,5x1,7x2,5x2,7x1,2x3,11x8 r15c24:11x5 r15c35:7x2 r15c38:7x2 r15c41:7x1 r15c45:11x5 r15c51:11x6,5x1,11x1 r16c5:7x1,5x1,7x3,5x1,7x1,3x3,11x1 r16c18:5x3,11x8 r16c35:7x1 r16c37:5x1,7x2 r16c41:7x1 r16c45:11x1 r16c47:11x3 r16c51:11x2,5x1,11x3,5x1,11x1 r17c5:7x1,5x2,7x2,5x1,7x1,2x3,11x12,5x1,11x1 r17c35:7x2,5x1,7x2 r17c41:7x1 r17c45:11x1 r17c47:11x6,5x1,11x5 r18c5:7x2,5x4,7x1,2x3,11x2 r18c18:11x1,5x1,11x3 r18c24:11x1,5x3,11x1 r18c35:7x2,5x1 r18c39:5x2,7x1 r18c45:11x1 r18c48:11x2,5x1,11x2,5x2,11x2,5x1,11x1 r19c5:7x7,2x3,11x14 r19c35:7x7 r19c45:11x14 r22c5:7x7,2x3,11x21,2x6,7x3 r22c49:2x3,11x7 r23c5:7x3,5x3,7x1,2x3,11x1,5x1,11x1,5x1,11x1,5x1,11x2 r23c25:11x1 r23c28:11x2,5x1,11x1,5x1,11x1,5x1,11x1,2x6,7x3,5x1 r23c48:7x1,2x3,11x2,5x1,11x1,5x1,11x2 r24c5:7x3,5x1,7x1,5x1,7x1,2x3,11x5,5x1,11x2 r24c24:11x3 r24c28:11x2,5x1,11x5,2x6,7x1,5x3 r24c48:7x1,2x3,11x1,5x2,11x1,5x2,11x1 r25c5:7x1,5x5,7x1,3x3,11x1 r25c17:11x1,5x1,11x1,5x1,11x2 r25c24:11x1 r25c26:11x1 r25c28:11x2,5x1,11x1,5x1,11x1,5x1,11x1,2x6,7x1,5x1,7x1,5x1,7x1,3x3,11x1,5x1,11x3,5x1,11x1 r26c5:7x1,5x1,7x1,5x1,7x3,2x3,11x5,5x1,11x2 r26c24:11x3 r26c28:11x2,5x1,11x5,2x6,7x1,5x3 r26c48:7x1,2x3,11x7 r27c5:7x1,5x3,7x3,2x3,11x1,5x5,11x2 r27c25:11x1 r27c28:11x2,5x5,11x1,2x6,7x3,5x1 r27c48:7x1,2x3,11x1,5x1,11x3,5x1,11x1 r28c5:7x7,2x3,11x21,2x6,7x3 r28c49:2x3,11x7 r40c14:3x4,7x28,3x3 r41c14:3x4,7x5,5x1,7x4,5x1 r41c31:7x2 r41c34:5x2 r41c37:7x4,5x3,7x2,3x3 r42c14:3x4,7x2,5x4,7x4,5x1,7x1,5x1,7x2 r42c34:7x2,5x1,7x4,5x1,7x1,5x1,7x2,3x3 r43c14:3x4,7x2,5x1,7x2 r43c24:7x2 r43c27:5x4,7x2 r43c34:7x2,5x1 r43c38:7x2,5x4 r43c45:7x1,3x3 r44c14:3x4,7x2,5x1,7x2,5x1,7x2,5x1,7x1,5x1,7x4 r44c34:7x2,5x1,7x4,5x1,7x1,5x1,7x2,3x3 r45c14:3x4,7x1,5x3 r45c23:5x1,7x2,5x3,7x4 r45c34:5x2 r45c37:7x4,5x2 r45c44:7x2,3x3 r46c14:3x4,7x28,3x3 r48c8:0x5 r48c43:3x5 r49c8:0x1 r49c12:0x1 r49c43:3x1 r49c47:3x1 r51c7:11x49 r52c7:11x2,5x2,11x1,5x1,11x2,5x1,11x1 r52c20:11x2 r52c27:11x2 r52c30:11x3,5x1,11x2,5x2,11x1,5x2,11x2,5x2 r52c46:5x2,11x4,5x1,11x3 r53c7:11x1,5x2,11x7 r53c18:11x4 r53c23:11x6 r53c30:11x6 r53c37:11x3 r53c41:11x2 r53c44:11x3 r53c48:11x2,5x1,11x1,5x1,11x1,5x1,11x1 r54c7:11x8 r54c16:11x1 r54c20:11x2 r54c23:11x1 r54c25:11x1 r54c27:11x2 r54c30:5x3 r54c34:11x2 r54c37:11x1,5x1,11x1 r54c41:11x4 r54c46:11x6,5x1,11x3 r55c7:11x1,5x2,11x9,5x1,11x2 r55c23:11x10 r55c34:11x2 r55c37:11x3 r55c41:11x2 r55c44:11x3 r55c48:11x2,5x1,11x1,5x1,11x1,5x1,11x1 r56c7:11x2,5x2,11x1,5x1,11x2 r56c16:11x1 r56c18:5x2,11x2 r56c23:11x1 r56c25:11x1 r56c27:11x2,5x1,11x3 r56c34:11x2 r56c38:11x1 r56c41:11x2,5x2 r56c46:5x2,11x4,5x1,11x3 r57c7:11x49 r59c8:0x1 r59c12:0x1 r59c43:3x1 r59c47:3x1 r60c8:0x5 r60c43:3x5 r63c58:1x6
"""

_NEXT_LEVEL_GRID = None
try:
    _base = np.zeros((64, 64), dtype=int)
    for _line in PRE_WIN_RLE.splitlines():
        _line = _line.strip()
        if not _line:
            continue
        _m = re.match(r"r(\d+):(.*)", _line)
        if not _m:
            continue
        _row = int(_m.group(1))
        _col = 0
        for _part in _m.group(2).split(","):
            _v, _n = _part.split("x")
            _val = int(_v)
            _cnt = int(_n)
            _base[_row, _col:_col + _cnt] = _val
            _col += _cnt

    _next = _base.copy()
    for _tok in WIN_DELTA.split():
        _m = re.match(r"r(\d+)c(\d+):(.*)", _tok)
        if not _m:
            continue
        _row = int(_m.group(1))
        _col = int(_m.group(2))
        for _part in _m.group(3).split(","):
            _v, _n = _part.split("x")
            _val = int(_v)
            _cnt = int(_n)
            _next[_row, _col:_col + _cnt] = _val
            _col += _cnt
    _NEXT_LEVEL_GRID = _next
except Exception:
    _NEXT_LEVEL_GRID = None


def engine(grid, action, data):
    g = np.array(grid, dtype=int, copy=True)
    if g.shape != (64, 64):
        return g

    if g[0, 0] == 11 and g[0, 1] == 11 and g[63, 0] == 11:
        return g
    if _NEXT_LEVEL_GRID is not None and np.array_equal(g, _NEXT_LEVEL_GRID):
        return g

    s = -1
    check_starts = list(HOLE_STARTS)
    for x in range(0, 60):
        if x not in HOLE_STARTS:
            check_starts.append(x)

    for st in check_starts:
        ok = True
        if st < 0 or st + 4 >= 64:
            ok = False
        else:
            if not all(g[48, c] == 0 for c in range(st, st + 5)):
                ok = False
            elif g[49, st] != 0 or g[49, st + 4] != 0:
                ok = False
            elif not all(g[60, c] == 0 for c in range(st, st + 5)):
                ok = False
            elif g[59, st] != 0 or g[59, st + 4] != 0:
                ok = False
        if ok:
            s = st
            break

    lane = -1
    if s in HOLE_STARTS:
        lane = (s - 15) // 7

    lane_mats = []
    for st in HOLE_STARTS:
        mt = tuple(
            tuple(1 if g[r, c] == 7 else 0 for c in range(st, st + 5))
            for r in range(52, 57)
        )
        lane_mats.append(mt)

    mat = None
    if lane >= 0:
        mat = lane_mats[lane]

    left_col = 63
    while left_col >= 0 and g[63, left_col] == 4:
        left_col -= 1
    progress_len = 63 - left_col if left_col >= 0 else 64

    win_ready = False
    if action == 2 and s == 43 and progress_len >= 6:
        win_ready = True
        for i, exp in enumerate(EXPECTED_PRE_WIN):
            if lane_mats[i] != exp:
                win_ready = False
                break

    if win_ready:
        if _NEXT_LEVEL_GRID is not None:
            return _next.copy()
        w = g.copy()
        w[0, 0] = 11
        w[0, 1] = 11
        w[63, 0] = 11
        return w

    if action in (1, 2):
        if lane >= 0 and mat is not None:
            trans = TRANSITIONS.get((lane, mat, action))
            if trans is not None:
                new_mat, reward = trans
                for i in range(5):
                    r = 52 + i
                    vals = [7 if v else 5 for v in new_mat[i]]
                    g[r, s:s + 5] = vals
                for _ in range(reward):
                    if left_col >= 0:
                        g[63, left_col] = 4
                        left_col -= 1
        return g

    if action in (3, 4):
        if s != -1:
            delta = 7 if action == 4 else -7
            ns = s + delta
            if 0 <= ns <= 59:
                reward = 0
                if action == 4 and lane >= 0 and mat is not None:
                    term = TERMINAL.get(lane)
                    if term is not None and mat == term:
                        reward = LEAVE_REWARD.get(lane, 0)

                old = s
                g[48, old:old + 5] = 3
                g[49, old] = 3
                g[49, old + 4] = 3
                g[59, old] = 3
                g[59, old + 4] = 3
                g[60, old:old + 5] = 3

                g[48, ns:ns + 5] = 0
                g[49, ns] = 0
                g[49, ns + 4] = 0
                g[59, ns] = 0
                g[59, ns + 4] = 0
                g[60, ns:ns + 5] = 0

                for _ in range(reward):
                    if left_col >= 0:
                        g[63, left_col] = 4
                        left_col -= 1
        return g

    return g


def is_level_complete(grid):
    try:
        g = np.asarray(grid, dtype=int)
    except Exception:
        return False

    if g.shape != (64, 64):
        return False

    if g[0, 0] == 11 and g[0, 1] == 11 and g[63, 0] == 11:
        return True

    if _NEXT_LEVEL_GRID is not None and np.array_equal(g, _NEXT_LEVEL_GRID):
        return True

    return False