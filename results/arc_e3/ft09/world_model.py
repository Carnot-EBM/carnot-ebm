import numpy as np


def _inside_toggle_block(arr, x, y):
    h, w = arr.shape
    if not (0 <= y < h and 0 <= x < w):
        return None
    color = int(arr[y, x])
    if color not in (8, 9, 14, 15):
        return None
    y0 = y
    while y0 > 0 and int(arr[y0 - 1, x]) == color:
        y0 -= 1
    y1 = y + 1
    while y1 < h and int(arr[y1, x]) == color:
        y1 += 1
    x0 = x
    while x0 > 0 and int(arr[y, x0 - 1]) == color:
        x0 -= 1
    x1 = x + 1
    while x1 < w and int(arr[y, x1]) == color:
        x1 += 1
    if (y1 - y0) < 2 or (x1 - x0) < 2:
        return None
    return y0, y1, x0, x1, color


def _toggle_color(color):
    if color == 8:
        return 9
    if color == 9:
        return 8
    if color == 14:
        return 15
    if color == 15:
        return 14
    return color


def engine(grid, action, data):
    out = np.array(grid, copy=True)
    if out.ndim != 2 or action != 6 or not isinstance(data, dict):
        return out
    # The rendered ft09 frame is a 2x display of a 32x32 puzzle plus a status row.
    # The offline transition sampler often clicks UI/status components; keep those no-op.
    x = int(data.get("x", -1))
    y = int(data.get("y", -1))
    if y >= out.shape[0] - 1:
        return out
    block = _inside_toggle_block(out, x, y)
    if block is None:
        return out
    y0, y1, x0, x1, color = block
    if (y1 - y0) > 8 or (x1 - x0) > 8:
        return out
    out[y0:y1, x0:x1] = _toggle_color(color)
    return out


def is_level_complete(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return False
    # Constraint tiles use colors 8/9/14/15; a completed level should have no visible
    # contradiction color pair in the central puzzle region. This is deliberately
    # conservative so planning cannot claim a solve from the initial render.
    core = arr[: max(0, arr.shape[0] - 1), :]
    return bool(
        np.count_nonzero(core == 14) == 0
        and np.count_nonzero(core == 15) == 0
        and np.any(core == 8)
    )
