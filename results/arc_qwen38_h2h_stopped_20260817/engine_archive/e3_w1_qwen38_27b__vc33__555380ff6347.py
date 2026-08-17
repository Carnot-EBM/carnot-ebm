import numpy as np


def engine(grid, action, data):
    g = np.array(grid, copy=True)
    if g.ndim != 2 or g.size == 0:
        return g

    try:
        act = int(action)
    except Exception:
        return g

    if act != 6 or data is None:
        return g

    try:
        x = int(data["x"])
        y = int(data["y"])
    except Exception:
        return g

    H, W = g.shape
    if not (0 <= y < H and 0 <= x < W):
        return g

    # The observed interactive trigger is a click on a 9-colored button/target.
    if g[y, x] != 9:
        return g

    TILE = 4

    def find_player():
        """Find the moving player's 11-colored key bar."""
        best = None
        for c in range(W - 1):
            mask = (g[:, c] == 11) & (g[:, c + 1] == 11)
            r = 0
            while r < H:
                if mask[r]:
                    r0 = r
                    while r < H and mask[r]:
                        r += 1
                    length = r - r0
                    if length >= 5:
                        adj = 0
                        rr = r0
                        while rr < r:
                            if c > 0 and g[rr, c - 1] == 4:
                                adj += 1
                            elif c > 1 and g[rr, c - 2] == 4:
                                adj += 1
                            rr += 1
                        if adj * 2 >= length:
                            cand = (r0, c, length, adj)
                            if best is None or r0 > best[0] or (r0 == best[0] and adj > best[3]):
                                best = cand
                else:
                    r += 1

        if best is not None:
            return best[0], best[1], best[2]

        # Fallback for the observed layout.
        if H >= 50 and W >= 52:
            for c in range(W - 1):
                if np.all(g[44:50, c:c + 2] == 11):
                    return 44, c, 6

        return None

    player = find_player()
    if player is None:
        return g

    R, K, PH = player

    def find_slot():
        """Find the static 11-colored slot embedded in a 5-colored bar above the player."""
        best = None
        for c in range(W - 1):
            mask = (g[:, c] == 11) & (g[:, c + 1] == 11)
            r = 0
            while r < H:
                if mask[r]:
                    r0 = r
                    while r < H and mask[r]:
                        r += 1
                    length = r - r0
                    if length >= 4 and r <= R:
                        embedded = False
                        rr = r0
                        while rr < r:
                            if c > 0 and c + 2 < W and g[rr, c - 1] == 5 and g[rr, c + 2] == 5:
                                embedded = True
                                break
                            rr += 1
                        if embedded:
                            cand = (r0, r - 1, c, length)
                            if best is None or r0 > best[0]:
                                best = cand
                else:
                    r += 1
        return best

    slot = find_slot()

    # If already aligned, do not move further.
    if slot is not None and K == slot[2]:
        return g

    newK = K - TILE
    if newK < 0:
        return g

    # Clear the old core strip left behind by the player.
    clear_top = (slot[1] + 1) if slot is not None else max(0, R - 12)
    clear_bottom = min(H - 1, R + PH - 1)
    c0 = max(0, K - 2)
    c1 = min(W - 1, K + 1)
    if clear_top <= clear_bottom and c0 <= c1:
        g[clear_top:clear_bottom + 1, c0:c1 + 1] = 0

    def paint(r, c, v):
        if 0 <= r < H and 0 <= c < W:
            g[r, c] = v

    # Draw the moved player shape.
    for i in range(PH):
        rr = R + i
        if not (0 <= rr < H):
            continue

        if PH >= 4 and (i < 2 or i >= PH - 2):
            body_cols = [newK - 2, newK - 1]
        else:
            body_cols = [newK - 4, newK - 3, newK - 2, newK - 1]

        for cc in body_cols:
            paint(rr, cc, 4)

        paint(rr, newK, 11)
        paint(rr, newK + 1, 11)

    # Extend the upper left-side 3-colored wall by one tile into empty space.
    if slot is not None:
        upper_top = 1
        upper_bottom = min(H - 1, max(upper_top, slot[0] - 1))
    else:
        upper_top = 1
        upper_bottom = min(H - 1, 27)

    for rr in range(upper_top, upper_bottom + 1):
        c = 0
        while c < W and g[rr, c] == 3:
            c += 1

        filled = 0
        cc = c
        while cc < W and filled < TILE:
            if g[rr, cc] == 0:
                g[rr, cc] = 3
                filled += 1
                cc += 1
            else:
                break

    # Update the top HUD progress marker at the right edge.
    old_w = 0
    c = W - 1
    while c >= 0 and g[0, c] == 4:
        old_w += 1
        c -= 1

    new_w = 1 if old_w == 0 else old_w + 2
    if new_w > W:
        new_w = W

    g[0, :] = 7
    if new_w > 0:
        g[0, W - new_w:] = 4

    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    if g.ndim != 2 or g.size == 0:
        return False

    H, W = g.shape

    runs = []
    for c in range(W - 1):
        mask = (g[:, c] == 11) & (g[:, c + 1] == 11)
        r = 0
        while r < H:
            if mask[r]:
                r0 = r
                while r < H and mask[r]:
                    r += 1
                length = r - r0
                if length >= 4:
                    runs.append((r0, r - 1, c, length))
            else:
                r += 1

    players = []
    for r0, r1, c, length in runs:
        if length < 5:
            continue
        adj = 0
        rr = r0
        while rr <= r1:
            if c > 0 and g[rr, c - 1] == 4:
                adj += 1
            elif c > 1 and g[rr, c - 2] == 4:
                adj += 1
            rr += 1
        if adj * 2 >= length:
            players.append((r0, r1, c, length))

    if not players:
        return False

    # Win when the player's key bar is horizontally aligned with the static slot.
    for pr0, pr1, pc, _ in players:
        for sr0, sr1, sc, sl in runs:
            if sc != pc or sl < 4 or sr1 >= pr0:
                continue

            embedded = False
            rr = sr0
            while rr <= sr1:
                if sc > 0 and sc + 2 < W and g[rr, sc - 1] == 5 and g[rr, sc + 2] == 5:
                    embedded = True
                    break
                rr += 1

            if embedded:
                return True

    return False