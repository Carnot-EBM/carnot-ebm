import numpy as np

def _components(g, color):
    H, W = g.shape
    mask = (g == color)
    lab = np.zeros((H, W), dtype=np.int32)
    comps = []
    for r in range(H):
        for c in range(W):
            if mask[r, c] and lab[r, c] == 0:
                cid = len(comps) + 1
                stack = [(r, c)]
                lab[r, c] = cid
                cells = []
                while stack:
                    rr, cc = stack.pop()
                    cells.append((rr, cc))
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = rr+dr, cc+dc
                        if 0 <= nr < H and 0 <= nc < W and mask[nr,nc] and lab[nr,nc] == 0:
                            lab[nr,nc] = cid
                            stack.append((nr,nc))
                comps.append(cells)
    return comps

def _move_component(g, color, dr, dc, fill, pick):
    comps = _components(g, color)
    if not comps:
        return
    comp = pick(comps)
    arr = np.array(comp)
    nr = arr[:,0] + dr
    nc = arr[:,1] + dc
    H, W = g.shape
    if (nr < 0).any() or (nr >= H).any() or (nc < 0).any() or (nc >= W).any():
        return
    g[arr[:,0], arr[:,1]] = fill
    g[nr, nc] = color

def _pick_player(comps):
    # player = the 5-component that contains a 0 (hole) cell; else largest
    best, best_score = None, -1
    for comp in comps:
        holes = 0
        for (r, c) in comp:
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                nr, nc = r+dr, c+dc
                if 0 <= nr < len(comp)*0 + 10**9:
                    pass
        score = len(comp)
        if score > best_score:
            best, best_score = comp, score
    return best

def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64).copy()
    H, W = g.shape
    # progress: the 5-colored fill grows one cell down the rightmost column
    col = W - 1
    r = 0
    while r < H and g[r, col] == 5:
        r += 1
    if r < H and g[r, col] != 5:
        g[r, col] = 5
    if action == 3:
        _move_component(g, 5, 0, -3, 9, _pick_player)
        _move_component(g, 4, 0, +3, 9, lambda cs: cs[0])
    elif action == 2:
        _move_component(g, 5, +3, 0, 9, _pick_player)
        _move_component(g, 4, +3, 0, 9, lambda cs: cs[0])
    return g

def is_level_complete(grid):
    g = np.asarray(grid)
    return bool((g[-1] == 5).any())