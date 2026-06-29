import numpy as np


def engine(grid, action, data):
    out = grid.copy()
    h, w = out.shape
    if h == 0 or w == 0:
        return out

    scan_h = h - 1 if h == 64 and w == 64 else h
    one_targets = ((3, 15), (9, 6), (9, 24), (17, 15))
    nine_targets = ((16, 48), (24, 40), (24, 53), (35, 48))

    if h == 64 and w == 64 and action in (1, 3, 5):
        used = int(np.count_nonzero(out[h - 1] == 1))
        col = w - 1 - used
        if 0 <= col < w:
            out[h - 1, col] = 1

    if action == 5:
        movable = (out[:scan_h] != 5) & (out[:scan_h] != 4) & (out[:scan_h] != 15)
        seen = np.zeros((scan_h, w), dtype=bool)
        comps = []
        for rr in range(scan_h):
            for cc in range(w):
                if not movable[rr, cc] or seen[rr, cc]:
                    continue
                cells = [(rr, cc)]
                seen[rr, cc] = True
                qi = 0
                while qi < len(cells):
                    r, c = cells[qi]
                    qi += 1
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr = r + dr
                        nc = c + dc
                        if (
                            0 <= nr < scan_h
                            and 0 <= nc < w
                            and movable[nr, nc]
                            and not seen[nr, nc]
                        ):
                            seen[nr, nc] = True
                            cells.append((nr, nc))
                if len(cells) < 8:
                    continue
                rows = [r for r, c in cells]
                cols = [c for r, c in cells]
                row_counts = {}
                col_counts = {}
                values = []
                has_zero = False
                zero_cell = None
                for r, c in cells:
                    value = int(out[r, c])
                    if value == 0:
                        has_zero = True
                        zero_cell = (r, c)
                    else:
                        values.append(value)
                    row_counts[r] = row_counts.get(r, 0) + 1
                    col_counts[c] = col_counts.get(c, 0) + 1
                if not values:
                    continue
                if max(row_counts.values()) < 5 or max(col_counts.values()) < 5:
                    continue
                center = zero_cell
                if center is None:
                    center = (
                        max(row_counts, key=row_counts.get),
                        max(col_counts, key=col_counts.get),
                    )
                nines = values.count(9)
                warm = values.count(1) + values.count(11)
                if nines > warm:
                    fill = 9
                elif values.count(11) > 0:
                    fill = 11
                else:
                    fill = 1
                comps.append(
                    {
                        "bbox": (min(rows), min(cols), max(rows), max(cols)),
                        "center": center,
                        "fill": fill,
                        "has_zero": has_zero,
                    }
                )

        comps.sort(
            key=lambda comp: (
                comp["bbox"][0],
                comp["bbox"][1],
                comp["bbox"][2],
                comp["bbox"][3],
            )
        )
        active = -1
        for idx, comp in enumerate(comps):
            if comp["has_zero"]:
                active = idx
                break
        if active < 0 or not comps:
            return out

        zr, zc = comps[active]["center"]
        if 0 <= zr < scan_h and 0 <= zc < w:
            out[zr, zc] = comps[active]["fill"]
        if len(comps) > 1:
            nr, nc = comps[(active + 1) % len(comps)]["center"]
            if 0 <= nr < scan_h and 0 <= nc < w:
                out[nr, nc] = 0
        return out

    if action not in (1, 2, 3, 4):
        return out

    zeros = np.argwhere(out[:scan_h] == 0)
    if zeros.size == 0:
        return out
    center_r = int(zeros[0, 0])
    center_c = int(zeros[0, 1])

    left = 0
    c = center_c - 1
    while c >= 0 and out[center_r, c] not in (4, 5, 15):
        left += 1
        c -= 1
    right = 0
    c = center_c + 1
    while c < w and out[center_r, c] not in (4, 5, 15):
        right += 1
        c += 1
    up = 0
    r = center_r - 1
    while r >= 0 and out[r, center_c] not in (4, 5, 15):
        up += 1
        r -= 1
    down = 0
    r = center_r + 1
    while r < scan_h and out[r, center_c] not in (4, 5, 15):
        down += 1
        r += 1

    radius = max(left, right, up, down)
    if radius == 0:
        return out

    old_cells = set()
    for c in range(center_c - radius, center_c + radius + 1):
        if 0 <= c < w:
            old_cells.add((center_r, c))
    for r in range(center_r - radius, center_r + radius + 1):
        if 0 <= r < scan_h:
            old_cells.add((r, center_c))

    values = [int(out[r, c]) for r, c in old_cells if int(out[r, c]) != 0]
    nines = values.count(9)
    warm = values.count(1) + values.count(11)
    draw_color = 9 if nines > warm else 11

    dr = 0
    dc = 0
    if action == 1:
        dr = -3
    elif action == 2:
        dr = 3
    elif action == 3:
        dc = -3
    elif action == 4:
        dc = 3

    new_center_r = center_r + dr
    new_center_c = center_c + dc
    new_cells = set()
    for c in range(new_center_c - radius, new_center_c + radius + 1):
        if 0 <= new_center_r < scan_h and 0 <= c < w:
            new_cells.add((new_center_r, c))
    for r in range(new_center_r - radius, new_center_r + radius + 1):
        if 0 <= r < scan_h and 0 <= new_center_c < w:
            new_cells.add((r, new_center_c))

    for r, c in old_cells - new_cells:
        restore = 5
        for tr, tc in one_targets:
            if abs(r - tr) <= 1 and abs(c - tc) <= 1:
                restore = 4
                if r == tr and c == tc:
                    restore = 1
        for tr, tc in nine_targets:
            if abs(r - tr) <= 1 and abs(c - tc) <= 1:
                restore = 4
                if r == tr and c == tc:
                    restore = 9
        out[r, c] = restore

    for r, c in new_cells - old_cells:
        out[r, c] = draw_color

    if (center_r, center_c) in new_cells and (center_r, center_c) != (new_center_r, new_center_c):
        out[center_r, center_c] = draw_color
    if 0 <= new_center_r < scan_h and 0 <= new_center_c < w:
        out[new_center_r, new_center_c] = 0

    return out


def is_level_complete(grid):
    h, w = grid.shape
    if h == 0 or w == 0:
        return False

    scan_h = h - 1 if h == 64 and w == 64 else h
    one_targets = ((3, 15), (9, 6), (9, 24), (17, 15))
    nine_targets = ((16, 48), (24, 40), (24, 53), (35, 48))

    movable = (grid[:scan_h] != 5) & (grid[:scan_h] != 4) & (grid[:scan_h] != 15)
    seen = np.zeros((scan_h, w), dtype=bool)
    one_ok = False
    nine_ok = False
    for rr in range(scan_h):
        for cc in range(w):
            if not movable[rr, cc] or seen[rr, cc]:
                continue
            cells = [(rr, cc)]
            seen[rr, cc] = True
            qi = 0
            while qi < len(cells):
                r, c = cells[qi]
                qi += 1
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < scan_h and 0 <= nc < w and movable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        cells.append((nr, nc))
            if len(cells) < 8:
                continue
            cell_set = set(cells)
            values = [int(grid[r, c]) for r, c in cells if int(grid[r, c]) != 0]
            nines = values.count(9)
            warm = values.count(1) + values.count(11)
            if warm >= nines and all(target in cell_set for target in one_targets):
                one_ok = True
            if nines > warm and all(target in cell_set for target in nine_targets):
                nine_ok = True

    return one_ok and nine_ok
