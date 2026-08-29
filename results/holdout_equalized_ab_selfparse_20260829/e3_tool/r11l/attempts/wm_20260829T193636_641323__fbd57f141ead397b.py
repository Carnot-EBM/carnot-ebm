import numpy as np

def _diamond_cells(cr, cc):
    cells = []
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) <= 2:
                cells.append((cr + dr, cc + dc))
    return cells

def engine(grid, action, data):
    g = np.array(grid, dtype=int).copy()
    if action != 6 or data is None:
        return g
    px, py = int(data.get('x', 0)), int(data.get('y', 0))
    H, W = g.shape
    # find the 5x5 diamond (white 15 ring with magenta 6 center)
    best = None
    for r in range(2, H - 2):
        for c in range(2, W - 2):
            if g[r, c] == 6:
                ok = True
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if abs(dr) + abs(dc) <= 2:
                            want = 6 if (dr == 0 and dc == 0) else 15
                            if g[r + dr, c + dc] != want:
                                ok = False
                if ok:
                    best = (r, c)
    if best is None:
        return g
    cr, cc = best
    # jump toward click: move along the dominant axis until within 1
    dr = np.sign(py - cr)
    dc = np.sign(px - cc)
    nr, nc = cr, cc
    # step in the axis with larger distance first, then the other
    steps = []
    if abs(py - cr) >= abs(px - cc):
        steps = [(int(dr), 0), (0, int(dc))]
    else:
        steps = [(0, int(dc)), (int(dr), 0)]
    for (sr, sc) in steps:
        if sr == 0 and sc == 0:
            continue
        # move as far as possible in this direction, stopping 1 short of target
        while True:
            nr2, nc2 = nr + sr, nc + sc
            if (sr != 0 and (nr2 == py or (sr > 0 and nr2 >= py) or (sr < 0 and nr2 <= py))):
                nr, nc = nr2, nc2
                break
            if (sc != 0 and (nc2 == px or (sc > 0 and nc2 >= px) or (sc < 0 and nc2 <= px))):
                nr, nc = nr2, nc2
                break
            nr, nc = nr2, nc2
    if (nr, nc) != (cr, cc) and 2 <= nr < H - 2 and 2 <= nc < W - 2:
        for (rr, ccc) in _diamond_cells(cr, cc):
            g[rr, ccc] = 5
        for (rr, ccc) in _diamond_cells(nr, nc):
            g[rr, ccc] = 6 if (rr == nr and ccc == nc) else 15
    return g

def is_level_complete(grid):
    return False
