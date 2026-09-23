import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    if action == 6 and data is not None:
        px, py = int(data['x']), int(data['y'])
        x, y = px // 1, py // 1
        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
            # Determine which slot was clicked (slots at rows 57-60)
            slots = [(18, 21), (26, 29), (34, 37), (42, 45)]
            target_color = None
            for sx0, sx1 in slots:
                if sx0 <= x <= sx1:
                    target_color = g[y][sx0]
                    break
            if target_color is not None:
                # Find the matching colored block in the top area (rows 1-6)
                found = False
                for ty in range(1, 7):
                    for tx in range(17, 46):
                        if g[ty][tx] == target_color:
                            # Check it's part of a valid block (not just background 5)
                            if target_color != 5:
                                # Move this block to the bottom slot
                                # Block is 4x4 centered around the color region
                                bx0, by0 = tx - 1, ty - 1
                                bx1, by1 = tx + 2, ty + 2
                                # Clear the source
                                for ry in range(max(0, by0), min(g.shape[0], by1 + 1)):
                                    for rx in range(max(0, bx0), min(g.shape[1], bx1 + 1)):
                                        if g[ry][rx] == target_color or g[ry][rx] == 5:
                                            g[ry][rx] = 4
                                # Place at destination
                                dx0, dy0 = sx0, 57
                                for ry in range(dy0, dy0 + 4):
                                    for rx in range(dx0, dx0 + 4):
                                        g[ry][rx] = target_color
                                found = True
                                break
                    if found:
                        break
                if not found:
                    # Maybe it's already placed; toggle (remove) from slot
                    for sx0, sx1 in slots:
                        if sx0 <= x <= sx1:
                            cur = g[57][sx0]
                            if cur != 4 and cur != 0:
                                for ry in range(57, 61):
                                    for rx in range(sx0, sx1 + 1):
                                        g[ry][rx] = 4
                            else:
                                # Try to place a new one - find matching color in top
                                pass
                            break
    return g


def is_level_complete(grid):
    # Check if all four bottom slots are filled with the correct colors
    slots = [(18, 21), (26, 29), (34, 37), (42, 45)]
    expected = [14, 15, 9, 11]
    for i, (sx0, sx1) in enumerate(slots):
        c = grid[57][sx0]
        if c != expected[i]:
            return False
    return True