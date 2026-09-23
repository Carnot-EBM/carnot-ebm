import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        x, y = px, py
        # Determine which slot was clicked (slots at cols 18-21, 26-29, 34-37, 42-45 in rows 57-60)
        slots = [(18, 21), (26, 29), (34, 37), (42, 45)]
        colors = [14, 15, 9, 11]
        target_slot = -1
        for i, (c0, c1) in enumerate(slots):
            if c0 <= x <= c1 and 57 <= y <= 60:
                target_slot = i
                break

        if target_slot >= 0:
            target_color = colors[target_slot]
            # Find the matching colored object in the top area (rows 0-7)
            found_row = -1
            found_col = -1
            for r in range(8):
                for c in range(grid.shape[1]):
                    if grid[r, c] == target_color:
                        found_row = r
                        found_col = c
                        break
                if found_row >= 0:
                    break

            if found_row >= 0:
                # Clear the slot (set to 0)
                for r in range(56, 62):
                    for c in range(slots[target_slot][0], slots[target_slot][1] + 1):
                        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
                            grid[r, c] = 0

                # Place the color in the slot
                for r in range(57, 61):
                    for c in range(slots[target_slot][0], slots[target_slot][1] + 1):
                        grid[r, c] = target_color

                # Remove from top area
                for r in range(8):
                    for c in range(grid.shape[1]):
                        if grid[r, c] == target_color:
                            grid[r, c] = 4
    return grid


def is_level_complete(grid):
    # Check if all four colored objects have been placed in their correct slots
    slots = [(18, 21), (26, 29), (34, 37), (42, 45)]
    colors = [14, 15, 9, 11]
    for i, (c0, c1) in enumerate(slots):
        expected = colors[i]
        for r in range(57, 61):
            for c in range(c0, c1 + 1):
                if grid[r, c] != expected:
                    return False
    return True