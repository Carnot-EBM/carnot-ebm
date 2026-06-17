import numpy as np


def engine(grid, action, data):
    out = np.array(grid, copy=True)
    if out.ndim != 2:
        return out

    h, w = out.shape
    if h == 0 or w == 0:
        return out

    if action in (1, 2, 3, 4, 5):
        for y in range(h):
            if out[y, w - 1] != 5:
                out[y, w - 1] = 5
                break
    else:
        return out

    if action == 5:
        return out

    boxes = []
    for color in (5, 4):
        visited = np.zeros((h, w), dtype=bool)
        best = None
        for sy in range(h):
            for sx in range(w):
                if visited[sy, sx] or out[sy, sx] != color:
                    continue
                stack = [(sy, sx)]
                visited[sy, sx] = True
                min_y = max_y = sy
                min_x = max_x = sx
                area = 0
                while stack:
                    y, x = stack.pop()
                    area += 1
                    if y < min_y:
                        min_y = y
                    if y > max_y:
                        max_y = y
                    if x < min_x:
                        min_x = x
                    if x > max_x:
                        max_x = x
                    for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and out[ny, nx] == color:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
                bh = max_y - min_y + 1
                bw = max_x - min_x + 1
                if bh > 1 and bw > 1 and (best is None or area > best[4]):
                    best = (min_y, max_y, min_x, max_x, area, color)
        if best is not None:
            boxes.append(best)

    if not boxes:
        return out

    moves = []
    for min_y, max_y, min_x, max_x, area, color in boxes:
        dy = 0
        dx = 0
        if action == 1:
            dy = -3
        elif action == 2:
            dy = 3
        elif action == 3:
            dx = -3 if color == 5 else 3
        elif action == 4:
            dx = 3 if color == 5 else -3

        nmin_y = min_y + dy
        nmax_y = max_y + dy
        nmin_x = min_x + dx
        nmax_x = max_x + dx
        if nmin_y < 0 or nmin_x < 0 or nmax_y >= h or nmax_x >= w:
            return out
        patch = out[min_y:max_y + 1, min_x:max_x + 1].copy()
        moves.append((min_y, max_y, min_x, max_x, nmin_y, nmax_y, nmin_x, nmax_x, patch))

    for min_y, max_y, min_x, max_x, nmin_y, nmax_y, nmin_x, nmax_x, patch in moves:
        out[min_y:max_y + 1, min_x:max_x + 1] = 9
    for min_y, max_y, min_x, max_x, nmin_y, nmax_y, nmin_x, nmax_x, patch in moves:
        out[nmin_y:nmax_y + 1, nmin_x:nmax_x + 1] = patch

    return out


def is_level_complete(grid):
    arr = np.asarray(grid)
    if arr.ndim != 2:
        return False

    h, w = arr.shape
    if h == 0 or w == 0:
        return False

    goal_colors = (1, 10, 11)
    visited = np.zeros((h, w), dtype=bool)
    for sy in range(h):
        for sx in range(w):
            if visited[sy, sx] or int(arr[sy, sx]) not in goal_colors:
                continue
            color = arr[sy, sx]
            stack = [(sy, sx)]
            visited[sy, sx] = True
            min_y = max_y = sy
            min_x = max_x = sx
            area = 0
            while stack:
                y, x = stack.pop()
                area += 1
                if y < min_y:
                    min_y = y
                if y > max_y:
                    max_y = y
                if x < min_x:
                    min_x = x
                if x > max_x:
                    max_x = x
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and arr[ny, nx] == color:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area > 1 and (max_y - min_y + 1) > 1 and (max_x - min_x + 1) > 1:
                return False

    return True
