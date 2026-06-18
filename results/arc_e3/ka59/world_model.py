import numpy as np


STEP = 3
PUSH_STEPS = 5
BACKGROUND = 1
PADDING = 2
TARGET = 4
WALL = 15
BLOCK_BORDER = 14
SELECTED = 0
UNSELECTED_INITIAL = 5
STEP_COUNTER_TOTAL = 100


def _blocks(grid):
    arr = np.asarray(grid)
    out = []
    h, w = arr.shape
    for cy in range(1, h - 1):
        for cx in range(1, w - 1):
            if int(arr[cy, cx]) not in (SELECTED, TARGET, UNSELECTED_INITIAL):
                continue
            y, x = cy - 1, cx - 1
            patch = arr[y : y + 3, x : x + 3]
            if (
                patch[0, 0] in (BLOCK_BORDER, SELECTED)
                and patch[0, 1] in (BLOCK_BORDER, SELECTED)
                and patch[0, 2] in (BLOCK_BORDER, SELECTED)
                and patch[1, 0] in (BLOCK_BORDER, SELECTED)
                and patch[1, 2] in (BLOCK_BORDER, SELECTED)
                and patch[2, 0] in (BLOCK_BORDER, SELECTED)
                and patch[2, 1] in (BLOCK_BORDER, SELECTED)
                and patch[2, 2] in (BLOCK_BORDER, SELECTED)
                and int(patch[1, 1]) in (SELECTED, TARGET, UNSELECTED_INITIAL)
                and (
                    (patch[0, 1] == BLOCK_BORDER and patch[2, 1] == BLOCK_BORDER)
                    or (patch[1, 0] == BLOCK_BORDER and patch[1, 2] == BLOCK_BORDER)
                )
            ):
                out.append((y, x, int(patch[1, 1])))
    return out


def _selected_block(grid):
    for y, x, center in _blocks(grid):
        if center == SELECTED:
            return y, x
    blocks = _blocks(grid)
    return (blocks[0][0], blocks[0][1]) if blocks else None


def _draw_block(out, y, x, center):
    out[y : y + 3, x : x + 3] = BLOCK_BORDER
    out[y + 1, x + 1] = center


def _normalize_block_borders(out):
    for y, x, center in _blocks(out):
        out[y : y + 3, x : x + 3] = BLOCK_BORDER
        out[y + 1, x + 1] = center


def _mark_selected_contacts(out):
    selected = _selected_block(out)
    if selected is None:
        return
    y, x = selected
    for by, bx, _center in _blocks(out):
        if by == y and bx == x:
            continue
        if by + 3 == y and bx < x + 3 and bx + 3 > x:
            out[y, x : x + 3] = SELECTED
        if y + 3 == by and bx < x + 3 and bx + 3 > x:
            out[y + 2, x : x + 3] = SELECTED
        if bx + 3 == x and by < y + 3 and by + 3 > y:
            out[y : y + 3, x] = SELECTED
        if x + 3 == bx and by < y + 3 and by + 3 > y:
            out[y : y + 3, x + 2] = SELECTED
    out[y + 1, x + 1] = SELECTED


def _refresh_block_marks(out):
    _normalize_block_borders(out)
    _mark_selected_contacts(out)


def _target_underlay(out, y, x):
    fill = np.full((3, 3), BACKGROUND, dtype=out.dtype)
    h, w = out.shape
    for ty in range(max(0, y - 5), min(h - 4, y + 2) + 1):
        for tx in range(max(0, x - 5), min(w - 4, x + 2) + 1):
            if not _is_target_border(out[ty : ty + 5, tx : tx + 5]):
                continue
            for yy in range(3):
                for xx in range(3):
                    gy, gx = y + yy, x + xx
                    inside = ty <= gy <= ty + 4 and tx <= gx <= tx + 4
                    on_border = gy in (ty, ty + 4) or gx in (tx, tx + 4)
                    if inside and on_border:
                        fill[yy, xx] = TARGET
    return fill


def _is_target_border(patch):
    if patch.shape != (5, 5):
        return False
    top = bool(np.all(patch[0, :] == TARGET))
    bottom = bool(np.all(patch[4, :] == TARGET))
    left = bool(np.all(patch[:, 0] == TARGET))
    right = bool(np.all(patch[:, 4] == TARGET))
    border = np.concatenate((patch[0, :], patch[4, :], patch[1:4, 0], patch[1:4, 4]))
    return bool((top or bottom) and (left or right) and np.count_nonzero(border == TARGET) >= 8)


def _erase_block(out, y, x):
    out[y : y + 3, x : x + 3] = _target_underlay(out, y, x)


def _rects_overlap(a_y, a_x, b_y, b_x):
    return a_y < b_y + 3 and a_y + 3 > b_y and a_x < b_x + 3 and a_x + 3 > b_x


def _blocked_static(out, y, x):
    h, w = out.shape
    if y < 0 or x < 0 or y + 3 > h - 1 or x + 3 > w:
        return True
    patch = out[y : y + 3, x : x + 3]
    return bool(np.any(np.isin(patch, [PADDING, WALL])))


def _move_hits_block(out, y, x, old_y, old_x):
    hits = []
    for by, bx, center in _blocks(out):
        if by == old_y and bx == old_x:
            continue
        if _rects_overlap(y, x, by, bx):
            hits.append((by, bx, center))
    return hits


def _can_place_after_erasing(out, y, x, old_y, old_x):
    if _blocked_static(out, y, x):
        return False
    temp = out.copy()
    _erase_block(temp, old_y, old_x)
    patch = temp[y : y + 3, x : x + 3]
    return bool(np.all(np.isin(patch, [BACKGROUND, TARGET])))


def _push_block(out, by, bx, center, dy, dx):
    ny = by + dy * STEP * PUSH_STEPS
    nx = bx + dx * STEP * PUSH_STEPS
    h, w = out.shape
    if ny < 0 or nx < 0 or ny + 3 > h - 1 or nx + 3 > w:
        return False
    _erase_block(out, by, bx)
    _draw_block(out, ny, nx, center)
    return True


def _tick(out):
    if out.ndim != 2 or out.shape[0] < 64 or out.shape[1] < 64:
        return
    row = out[63]
    filled_count = int(np.count_nonzero(row == TARGET))
    if filled_count <= 0:
        return
    if filled_count >= 64:
        next_count = round(64 * (STEP_COUNTER_TOTAL - 1) / STEP_COUNTER_TOTAL)
    else:
        max_hidden_steps = int((filled_count + 0.5) * STEP_COUNTER_TOTAL / 64)
        next_count = round(64 * max(max_hidden_steps - 1, 0) / STEP_COUNTER_TOTAL)
    next_count = max(0, min(64, int(next_count)))
    row[:] = SELECTED
    if next_count:
        row[:next_count] = TARGET


def _click_select(out, data):
    if not data:
        return
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))
    for by, bx, _center in _blocks(out):
        if by <= y <= by + 2 and bx <= x <= bx + 2:
            current = _selected_block(out)
            if current is not None:
                out[current[0] + 1, current[1] + 1] = TARGET
            out[by + 1, bx + 1] = SELECTED
            _refresh_block_marks(out)
            return


def engine(grid, action, data):
    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out

    if action == 6:
        _tick(out)
        _click_select(out, data)
        return out

    if action not in (1, 2, 3, 4):
        return out

    _tick(out)
    selected = _selected_block(out)
    if selected is None:
        return out
    y, x = selected
    dy, dx = {
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }[action]
    ny, nx = y + dy * STEP, x + dx * STEP

    hits = _move_hits_block(out, ny, nx, y, x)
    if hits:
        for by, bx, center in hits:
            _push_block(out, by, bx, center, dy, dx)
        _refresh_block_marks(out)
        return out

    if not _can_place_after_erasing(out, ny, nx, y, x):
        _refresh_block_marks(out)
        return out
    _erase_block(out, y, x)
    _draw_block(out, ny, nx, SELECTED)
    _refresh_block_marks(out)
    return out


def is_level_complete(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return False
    occupied = []
    for by, bx, _center in _blocks(arr):
        border = arr[max(0, by - 1) : by + 4, max(0, bx - 1) : bx + 4]
        if border.shape == (5, 5) and np.count_nonzero(border == TARGET) >= 12:
            occupied.append((by, bx))
    return len(set(occupied)) >= 2
