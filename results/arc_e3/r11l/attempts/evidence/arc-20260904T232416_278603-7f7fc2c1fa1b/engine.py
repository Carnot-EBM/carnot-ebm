import numpy as np


def engine(grid, action, data):
    g = np.array(grid, dtype=np.int64, copy=True)
    if g.ndim != 2:
        return g

    H, W = g.shape
    FLOOR = 5
    MOVE_COLORS = (6, 12, 14, 15)
    SWITCH_COLORS = (3, 12, 14, 15)
    dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

    if action in dirs:
        dr, dc = dirs[action]
        cores = np.argwhere(g == 6)
        sprite_masks = {}

        for r, c in cores:
            r = int(r)
            c = int(c)

            r0 = max(0, r - 2)
            r1 = min(H - 1, r + 2)
            c0 = max(0, c - 2)
            c1 = min(W - 1, c + 2)
            sub = g[r0:r1 + 1, c0:c1 + 1]

            cnt15 = int(np.sum(sub == 15))
            cnt14 = int(np.sum(sub == 14))
            cnt12 = int(np.sum(sub == 12))
            total = int(np.sum((sub == 6) | (sub == 12) | (sub == 14) | (sub == 15)))

            if max(cnt15, cnt14, cnt12) >= 10 and total >= 12:
                mask = np.zeros((H, W), dtype=bool)
                stack = [(r, c)]
                mask[r, c] = True
                R = 3

                while stack:
                    rr, cc = stack.pop()
                    for nr in range(rr - 1, rr + 2):
                        for nc in range(cc - 1, cc + 2):
                            if nr == rr and nc == cc:
                                continue
                            if 0 <= nr < H and 0 <= nc < W and not mask[nr, nc]:
                                if abs(nr - r) <= R and abs(nc - c) <= R:
                                    v = int(g[nr, nc])
                                    if v in MOVE_COLORS:
                                        mask[nr, nc] = True
                                        stack.append((nr, nc))

                coords = np.argwhere(mask)
                if len(coords) >= 5:
                    sprite_masks[(r, c)] = mask

        if not sprite_masks:
            return g

        ng = g.copy()

        for cr, cc in sorted(sprite_masks.keys()):
            mask = sprite_masks[(cr, cc)]
            old_coords = np.argwhere(mask)
            old_set = set((int(a), int(b)) for a, b in old_coords)
            old_vals = {}
            for a, b in old_coords:
                old_vals[(int(a), int(b))] = int(g[a, b])

            core_dest_r = cr + dr
            core_dest_c = cc + dc
            exiting = (core_dest_r < 0 or core_dest_r >= H or
                       core_dest_c < 0 or core_dest_c >= W)

            blocked = False
            new_positions = []

            for a, b in old_coords:
                nr = int(a) + dr
                nc = int(b) + dc

                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    if not exiting:
                        blocked = True
                        break
                    continue

                if (nr, nc) not in old_set and int(ng[nr, nc]) != FLOOR:
                    blocked = True
                    break

                new_positions.append(((int(a), int(b)), (nr, nc)))

            if blocked:
                continue

            for a, b in old_coords:
                ng[int(a), int(b)] = FLOOR

            if not exiting:
                for (ar, ac), (nr, nc) in new_positions:
                    ng[nr, nc] = old_vals[(ar, ac)]

        return ng

    elif action == 6:
        px = None
        py = None
        if isinstance(data, dict):
            px = data.get("x")
            py = data.get("y")

        try:
            c = int(px)
            r = int(py)
        except Exception:
            return g

        if 0 <= r < H and 0 <= c < W:
            v = int(g[r, c])

            if v == 0 and (r == 0 or r == H - 1 or c == 0 or c == W - 1):
                g[r, c] = FLOOR
                return g

            if v in SWITCH_COLORS:
                opened = False
                for rr in range(H):
                    for cc in range(W):
                        if (rr == 0 or rr == H - 1 or cc == 0 or cc == W - 1):
                            if int(g[rr, cc]) == 0:
                                g[rr, cc] = FLOOR
                                opened = True
                                break
                    if opened:
                        break

        return g

    else:
        return g


def is_level_complete(grid):
    g = np.asarray(grid, dtype=np.int64)
    if g.ndim != 2 or g.size == 0:
        return False

    H, W = g.shape
    if not np.any(g != 0):
        return False

    cores = np.argwhere(g == 6)

    for r, c in cores:
        r = int(r)
        c = int(c)

        r0 = max(0, r - 2)
        r1 = min(H - 1, r + 2)
        c0 = max(0, c - 2)
        c1 = min(W - 1, c + 2)
        sub = g[r0:r1 + 1, c0:c1 + 1]

        cnt15 = int(np.sum(sub == 15))
        cnt14 = int(np.sum(sub == 14))
        cnt12 = int(np.sum(sub == 12))
        total = int(np.sum((sub == 6) | (sub == 12) | (sub == 14) | (sub == 15)))

        if max(cnt15, cnt14, cnt12) >= 10 and total >= 12:
            return False

    return True