import numpy as np


def engine(grid, action, data):
    out = grid.copy()
    h, w = out.shape
    if h == 0 or w == 0:
        return out

    status_row = h - 1 if h == 64 and w == 64 else -1
    target_a = 11 if np.any(out == 11) else 1
    target_groups = (
        (target_a, ((3, 15), (9, 6), (9, 24), (17, 15))),
        (9, ((16, 48), (24, 40), (24, 53), (35, 48))),
    )

    if status_row >= 0:
        last = out[status_row]
        if np.all((last == 15) | (last == 1)):
            remaining = int(np.count_nonzero(last == 15))
            remaining = max(0, remaining - 1)
            out[status_row, :remaining] = 15
            out[status_row, remaining:] = 1
        elif action != 1:
            used = np.flatnonzero(last == 1)
            col = w - 1 - len(used)
            if 0 <= col < w:
                out[status_row, col] = 1

    if action not in (1, 2, 3, 4, 5):
        return out

    scan_h = h - 1 if status_row >= 0 else h
    movable = np.zeros((h, w), dtype=bool)
    if scan_h > 0:
        movable[:scan_h] = (out[:scan_h] != 5) & (out[:scan_h] != 4) & (out[:scan_h] != 15)

    seen = np.zeros((h, w), dtype=bool)
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
                    if 0 <= nr < scan_h and 0 <= nc < w and movable[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        cells.append((nr, nc))
            if len(cells) >= 6:
                vals = [int(out[r, c]) for r, c in cells if out[r, c] != 0]
                if vals:
                    color = max(set(vals), key=vals.count)
                    rows = [r for r, c in cells]
                    cols = [c for r, c in cells]
                    comps.append(
                        {
                            "cells": cells,
                            "color": int(color),
                            "bbox": (min(rows), min(cols), max(rows), max(cols)),
                            "has_zero": any(out[r, c] == 0 for r, c in cells),
                        }
                    )

    if not comps:
        return out

    comps.sort(key=lambda comp: (comp["bbox"][0], comp["bbox"][1], comp["color"], len(comp["cells"])))
    active_index = -1
    for idx, comp in enumerate(comps):
        if comp["has_zero"]:
            active_index = idx
            break
    if active_index < 0:
        return out

    if action == 5:
        for r, c in comps[active_index]["cells"]:
            if out[r, c] == 0:
                out[r, c] = comps[active_index]["color"]
                break
        next_comp = comps[(active_index + 1) % len(comps)]
        row_counts = {}
        col_counts = {}
        for r, c in next_comp["cells"]:
            row_counts[r] = row_counts.get(r, 0) + 1
            col_counts[c] = col_counts.get(c, 0) + 1
        center_r = max(row_counts, key=row_counts.get)
        center_c = max(col_counts, key=col_counts.get)
        if 0 <= center_r < scan_h and 0 <= center_c < w:
            out[center_r, center_c] = 0
        return out

    dy = 0
    dx = 0
    if action == 1:
        dy = -3
    elif action == 2:
        dy = 3
    elif action == 3:
        dx = -3
    elif action == 4:
        dx = 3

    zero_pos = np.argwhere(out[:scan_h] == 0)
    if zero_pos.size == 0:
        return out
    center_r = int(zero_pos[0, 0])
    center_c = int(zero_pos[0, 1])
    color = comps[active_index]["color"]

    left = 0
    c = center_c - 1
    while c >= 0 and out[center_r, c] in (color, 0):
        left += 1
        c -= 1
    right = 0
    c = center_c + 1
    while c < w and out[center_r, c] in (color, 0):
        right += 1
        c += 1
    up = 0
    r = center_r - 1
    while r >= 0 and out[r, center_c] in (color, 0):
        up += 1
        r -= 1
    down = 0
    r = center_r + 1
    while r < scan_h and out[r, center_c] in (color, 0):
        down += 1
        r += 1

    half_x = max(left, right)
    half_y = max(up, down)
    if half_x == 0 or half_y == 0:
        cells = comps[active_index]["cells"]
    else:
        cells = []
        for c in range(center_c - half_x, center_c + half_x + 1):
            cells.append((center_r, c))
        for r in range(center_r - half_y, center_r + half_y + 1):
            if r != center_r:
                cells.append((r, center_c))

    for r, c in cells:
        if 0 <= r < scan_h and 0 <= c < w:
            restore = 5
            for t_color, centers in target_groups:
                for tr, tc in centers:
                    if abs(r - tr) <= 1 and abs(c - tc) <= 1:
                        restore = 4
                        if r == tr and c == tc:
                            restore = t_color
            out[r, c] = restore

    new_center_r = center_r + dy
    new_center_c = center_c + dx
    for r, c in cells:
        nr = r + dy
        nc = c + dx
        if 0 <= nr < scan_h and 0 <= nc < w:
            out[nr, nc] = color
    if 0 <= new_center_r < scan_h and 0 <= new_center_c < w:
        out[new_center_r, new_center_c] = 0
    return out


def is_level_complete(grid):
    h, w = grid.shape
    if h < 64 or w < 64:
        return False

    scan_h = h - 1 if h == 64 and w == 64 else h
    target_a = 11 if np.any(grid == 11) else 1
    target_groups = (
        (target_a, ((3, 15), (9, 6), (9, 24), (17, 15))),
        (9, ((16, 48), (24, 40), (24, 53), (35, 48))),
    )

    movable = np.zeros((h, w), dtype=bool)
    movable[:scan_h] = (grid[:scan_h] != 5) & (grid[:scan_h] != 4) & (grid[:scan_h] != 15)
    seen = np.zeros((h, w), dtype=bool)
    covered = {target_a: set(), 9: set()}
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
            if len(cells) < 6:
                continue
            vals = [int(grid[r, c]) for r, c in cells if grid[r, c] != 0]
            if not vals:
                continue
            color = max(set(vals), key=vals.count)
            cell_set = set(cells)
            for t_color, centers in target_groups:
                if color == t_color:
                    for center in centers:
                        if center in cell_set:
                            covered[t_color].add(center)

    for t_color, centers in target_groups:
        if any(center not in covered[t_color] for center in centers):
            return False
    return True
