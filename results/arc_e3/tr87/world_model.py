import numpy as np


def greedy_rewrite(seq, rules):
    """Apply tr87's left-to-right first-prefix rewrite once."""
    out = []
    pos = 0
    seq = tuple(seq)
    normalized = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
    while pos < len(seq):
        for lhs, rhs in normalized:
            if seq[pos : pos + len(lhs)] == lhs:
                out.extend(rhs)
                pos += len(lhs)
                break
        else:
            return None
    return tuple(out)


def rewrite_passes(seq, rules, passes):
    """Apply the greedy tr87 rewrite for one or more passes."""
    cur = tuple(seq)
    for _ in range(passes):
        cur = greedy_rewrite(cur, rules)
        if cur is None:
            return None
    return cur


def transition_fixture():
    """Executable fixture for the localized two-pass rewrite mismatch."""
    rules = [
        (("A4",), ("B3",)),
        (("B3",), ("C6", "C1")),
    ]
    expected = ("C6", "C1")
    observed = rewrite_passes(("A4",), rules, 2)
    return {
        "transition": "tr87:L7:two_pass_greedy_rewrite",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
    }


def _advance_timer(out):
    if out.ndim != 2 or out.shape[0] == 0:
        return out
    row = out.shape[0] - 1
    vals = set(int(v) for v in out[row].tolist())
    if 1 not in vals:
        return out
    for col in range(out.shape[1] - 1, -1, -1):
        if int(out[row, col]) == 1:
            out[row, col] = 4
            break
    return out


def _marker_columns(arr):
    if arr.ndim != 2:
        return []
    h, w = arr.shape
    cols = []
    for x in range(w - 4):
        top = arr[max(0, h - 16) : max(0, h - 14), x : x + 5]
        bottom = arr[max(0, h - 5) : max(0, h - 3), x : x + 5]
        top_has_marker = np.count_nonzero(top == 0) >= 6
        bottom_has_marker = np.count_nonzero(bottom == 0) >= 6
        if top.size and bottom.size and top_has_marker and bottom_has_marker:
            cols.append(x)
    return cols


def _move_selection(out, direction):
    cols = _marker_columns(out)
    if not cols:
        return out
    x0 = cols[0]
    step = 7 * direction
    x1 = max(0, min(out.shape[1] - 5, x0 + step))
    y0 = max(0, out.shape[0] - 16)
    y1 = max(0, out.shape[0] - 3)
    old = out[y0:y1, x0 : x0 + 5] == 0
    out[y0:y1, x0 : x0 + 5][old] = 3
    out[y0:y1, x1 : x1 + 5] = np.where(out[y0:y1, x1 : x1 + 5] == 3, 0, out[y0:y1, x1 : x1 + 5])
    return out


def _cycle_selected_piece(out, direction):
    cols = _marker_columns(out)
    if not cols:
        return out
    x0 = cols[0]
    y0 = max(0, out.shape[0] - 12)
    patch = out[y0 : min(out.shape[0], y0 + 5), x0 : min(out.shape[1], x0 + 5)]
    mask = (patch == 5) | (patch == 7)
    if not np.any(mask):
        return out
    patch[mask] = (
        np.where(patch[mask] == 5, 7, 5)
        if direction > 0
        else np.where(patch[mask] == 7, 5, 7)
    )
    return out


def engine(grid, action, data):
    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out
    if action == 3:
        _move_selection(out, -1)
    elif action == 4:
        _move_selection(out, 1)
    elif action == 1:
        _cycle_selected_piece(out, -1)
    elif action == 2:
        _cycle_selected_piece(out, 1)
    if action in (1, 2, 3, 4):
        _advance_timer(out)
    return out


def is_level_complete(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return False
    return bool(np.any(arr == 10) and np.any(arr == 11) and np.count_nonzero(arr[-1] == 4) > 8)
