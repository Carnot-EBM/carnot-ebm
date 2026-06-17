import numpy as np


STEP = 3
BACKGROUND = 1
PADDING = 2
TARGET = 4
WALL = 15
BLOCK_BORDER = 14
SELECTED = 0
UNSELECTED_INITIAL = 5


def _blocks(grid):
    arr = np.asarray(grid)
    out = []
    h, w = arr.shape
    for y in range(0, h - 2):
        for x in range(0, w - 2):
            patch = arr[y : y + 3, x : x + 3]
            if (
                patch[0, 0] == BLOCK_BORDER
                and patch[0, 1] == BLOCK_BORDER
                and patch[0, 2] == BLOCK_BORDER
                and patch[1, 0] == BLOCK_BORDER
                and patch[1, 2] == BLOCK_BORDER
                and patch[2, 0] == BLOCK_BORDER
                and patch[2, 1] == BLOCK_BORDER
                and patch[2, 2] == BLOCK_BORDER
                and int(patch[1, 1]) in (SELECTED, TARGET, UNSELECTED_INITIAL)
            ):
                out.append((y, x, int(patch[1, 1])))
    return out


def _selected_block(grid):
    for y, x, center in _blocks(grid):
        if center == SELECTED:
            return y, x
    blocks = _blocks(grid)
    return (blocks[0][0], blocks[0][1]) if blocks else None


def _restore_center(out, y, x):
    out[y + 1, x + 1] = TARGET


def _draw_block(out, y, x, center):
    out[y : y + 3, x : x + 3] = BLOCK_BORDER
    out[y + 1, x + 1] = center


def _underlay_for(out, y, x):
    patch = out[y : y + 3, x : x + 3]
    if np.any(patch == TARGET):
        return TARGET
    return BACKGROUND


def _erase_block(out, y, x):
    fill = _underlay_for(out, y, x)
    out[y : y + 3, x : x + 3] = fill


def _blocked(out, y, x, old_y, old_x):
    h, w = out.shape
    if y < 0 or x < 0 or y + 3 > h - 1 or x + 3 > w:
        return True
    patch = out[y : y + 3, x : x + 3].copy()
    old = out[old_y : old_y + 3, old_x : old_x + 3].copy()
    out[old_y : old_y + 3, old_x : old_x + 3] = _underlay_for(out, old_y, old_x)
    allowed = np.isin(out[y : y + 3, x : x + 3], [BACKGROUND, TARGET])
    out[old_y : old_y + 3, old_x : old_x + 3] = old
    return not bool(np.all(allowed)) or bool(np.any(np.isin(patch, [PADDING, WALL])))


def _tick(out):
    if out.ndim != 2 or out.shape[0] < 64 or out.shape[1] < 64:
        return
    row = out[63]
    filled = np.flatnonzero(row == TARGET)
    if filled.size:
        row[int(filled[-1])] = SELECTED


def _click_select(out, data):
    if not data:
        return
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))
    for by, bx, _center in _blocks(out):
        if by <= y <= by + 2 and bx <= x <= bx + 2:
            current = _selected_block(out)
            if current is not None:
                _restore_center(out, current[0], current[1])
            out[by + 1, bx + 1] = SELECTED
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
        1: (-STEP, 0),
        2: (STEP, 0),
        3: (0, -STEP),
        4: (0, STEP),
    }[action]
    ny, nx = y + dy, x + dx
    if _blocked(out, ny, nx, y, x):
        return out
    _erase_block(out, y, x)
    _draw_block(out, ny, nx, SELECTED)
    return out


def is_level_complete(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return False
    blocks = _blocks(arr)
    occupied = []
    for by, bx, _center in blocks:
        border = arr[max(0, by - 1) : by + 4, max(0, bx - 1) : bx + 4]
        if border.shape == (5, 5) and np.count_nonzero(border == TARGET) >= 12:
            occupied.append((by, bx))
    return len(set(occupied)) >= 2
