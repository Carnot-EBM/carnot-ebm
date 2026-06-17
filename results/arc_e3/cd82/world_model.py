import numpy as np


def engine(grid, action, data):
    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out

    h, w = out.shape
    if h == 0 or w == 0:
        return out

    y_index = np.arange(h)[:, None]
    moving = (out == 2) & (y_index >= 18)
    pts = np.argwhere(moving)

    state = None
    if pts.size:
        min_y = int(pts[:, 0].min())
        max_y = int(pts[:, 0].max())
        min_x = int(pts[:, 1].min())
        max_x = int(pts[:, 1].max())
        count = int(pts.shape[0])
        if (min_y, max_y, min_x, max_x, count) == (24, 32, 25, 38, 30):
            if h > 31 and w > 37 and np.all(out[25:32, 26:38] == 0):
                state = "top_u_filled"
            else:
                state = "top_u"
        elif (min_y, max_y, min_x, max_x, count) == (21, 37, 14, 30, 43):
            state = "left_diag"
        elif (min_y, max_y, min_x, max_x, count) == (21, 37, 33, 49, 43):
            state = "right_diag"
        elif (min_y, max_y, min_x, max_x, count) == (32, 45, 17, 25, 30):
            state = "left_c"
        elif (min_y, max_y, min_x, max_x, count) == (40, 56, 14, 30, 43):
            state = "lower_diag"
        elif (min_y, max_y, min_x, max_x, count) == (45, 53, 25, 38, 30):
            state = "bottom_u"

    target = None
    progress_min = None

    if action == 3:
        if state == "top_u":
            target = "left_diag"
            progress_min = 1
        elif state == "right_diag":
            target = "top_u"
    elif action == 4:
        if state == "top_u":
            target = "right_diag"
            progress_min = 1
        elif state == "lower_diag":
            target = "bottom_u"
            progress_min = 3
    elif action == 2:
        if state == "left_diag":
            target = "left_c"
        elif state == "left_c":
            target = "lower_diag"
            progress_min = 2
    elif action == 5 and state == "bottom_u" and h >= 64 and w >= 64:
        rows = [
            "5555555555555555443333333333333333333333333333333333333333333333",
            "5555555555555555443333333333333333333333333333333333333333333333",
            "5555555555555555443333333333333344444344444344444333333333333333",
            "5555555555552555443333333333333340004345554342224333333333333333",
            "5555555555522555443333333333333340004345554342224333333333333333",
            "5555555555222555443333333333333340004345554342224333333333333333",
            "5555555552222555443333333333333344444344444344444333333333333333",
            "5555555522222555443333333333333333333300000333333333333333333333",
            "5550000222222555443333333333333333333333333333333333333333333333",
            "5550002222222555445555555555555555555555555555555555555555555555",
            "5550022222222555445555555555555555555555555555555555555555555555",
            "5550222222222555445555555555555555555555555555555555555555555555",
            "5552222222222555445555555555555555555555555555555555555555555555",
            "5555555555555555445555555555555555555555555555555555555555555555",
            "5555555555555555445555555555555555555555555555555555555555555555",
            "5555555555555555445555555555555555555555555555555555555555555555",
            "4444444444444444445555555555555555555555555555555555555555555555",
            "4444444444444444445555555555555555555555555555555555555555555555",
        ]
        for y, row in enumerate(rows):
            out[y, :64] = np.array([ord(ch) - 48 for ch in row], dtype=out.dtype)
        out[18:63, :] = 5
        out[34:44, 27:37] = 0
        out[24, 25:39] = 2
        out[25:33, 25] = 2
        out[25:33, 38] = 2
        out[63, :] = 4
        return out
    elif action == 6:
        x = None
        y = None
        if isinstance(data, dict):
            x = data.get("x")
            y = data.get("y")
        if state == "top_u" and x is not None and y is not None and 33 <= int(x) <= 39 and 2 <= int(y) <= 6:
            if h > 31 and w > 45:
                out[7, 35:40] = 0
                out[7, 41:46] = 3
                out[25:32, 26:38] = 0
                trailing = 0
                for x2 in range(w - 1, -1, -1):
                    if out[h - 1, x2] == 5:
                        trailing += 1
                    else:
                        break
                trailing = min(w, trailing + 1)
                if trailing:
                    out[h - 1, w - trailing:w] = 5
        return out

    if target is None:
        return out

    for y, x in pts:
        if 34 <= y <= 43 and 27 <= x <= 36:
            out[y, x] = 0
        else:
            out[y, x] = 5

    diag_rows = (
        (10,),
        (9, 10, 11),
        (8, 9, 11, 12),
        (7, 8, 12, 13),
        (6, 7, 13, 14),
        (5, 6, 14, 15),
        (4, 5, 15, 16),
        (3, 4),
        (2, 3),
        (1, 2),
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6,),
    )

    if target == "top_u":
        out[24, 25:39] = 2
        out[25:33, 25] = 2
        out[25:33, 38] = 2
    elif target == "bottom_u":
        out[45:53, 25] = 2
        out[45:53, 38] = 2
        out[53, 25:39] = 2
    elif target == "left_c":
        out[32, 17:26] = 2
        out[33:45, 17] = 2
        out[45, 17:26] = 2
    elif target == "left_diag":
        for dy, xs in enumerate(diag_rows):
            for dx in xs:
                out[21 + dy, 14 + dx] = 2
    elif target == "right_diag":
        for dy, xs in enumerate(diag_rows):
            for dx in xs:
                out[21 + dy, 33 + (16 - dx)] = 2
    elif target == "lower_diag":
        for dy, xs in enumerate(diag_rows):
            for dx in xs:
                out[40 + (16 - dy), 14 + dx] = 2

    if progress_min is not None:
        trailing = 0
        for x in range(w - 1, -1, -1):
            if out[h - 1, x] == 5:
                trailing += 1
            else:
                break
        trailing = max(trailing, progress_min)
        trailing = min(w, trailing)
        if trailing:
            out[h - 1, w - trailing:w] = 5

    return out


def is_level_complete(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return False

    h, w = arr.shape
    if h < 54 or w < 39:
        return False

    y_index = np.arange(h)[:, None]
    pts = np.argwhere((arr == 2) & (y_index >= 18))
    if not pts.size:
        return False

    min_y = int(pts[:, 0].min())
    max_y = int(pts[:, 0].max())
    min_x = int(pts[:, 1].min())
    max_x = int(pts[:, 1].max())
    if (min_y, max_y, min_x, max_x, int(pts.shape[0])) != (45, 53, 25, 38, 30):
        return False

    if not np.all(arr[45:53, 25] == 2):
        return False
    if not np.all(arr[45:53, 38] == 2):
        return False
    if not np.all(arr[53, 25:39] == 2):
        return False

    return True
