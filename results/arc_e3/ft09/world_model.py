import numpy as np


def _toggle_component(arr, x, y):
    h, w = arr.shape
    if not (0 <= y < h and 0 <= x < w):
        return None
    color = int(arr[y, x])
    if color not in (8, 9, 14, 15):
        return None
    stack = [(y, x)]
    seen = {(y, x)}
    cells = []
    while stack:
        cy, cx = stack.pop()
        cells.append((cy, cx))
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and int(arr[ny, nx]) == color:
                seen.add((ny, nx))
                stack.append((ny, nx))
    rows = [cell[0] for cell in cells]
    cols = [cell[1] for cell in cells]
    y0, y1 = min(rows), max(rows) + 1
    x0, x1 = min(cols), max(cols) + 1
    if len(cells) < 4 or (y1 - y0) < 2 or (x1 - x0) < 2:
        return None
    if (y1 - y0) > 8 or (x1 - x0) > 8:
        return None
    return cells, color


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
    component = _toggle_component(out, x, y)
    if component is None:
        return out
    cells, color = component
    for cy, cx in cells:
        out[cy, cx] = _toggle_color(color)
    return out


def transition_fixture():
    before = np.zeros((8, 8), dtype=int)
    before[2:5, 2:5] = 8
    before[2, 4] = 0
    before[6, 6] = 8
    expected = before.copy()
    mask = expected[2:5, 2:5] == 8
    expected[2:5, 2:5][mask] = 9
    observed = engine(before, 6, {"x": 3, "y": 3})
    return {
        "transition": "ft09:L2:component_click_residual",
        "expected": expected.tolist(),
        "observed": observed.tolist(),
        "passed": bool(np.array_equal(observed, expected)),
    }


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
