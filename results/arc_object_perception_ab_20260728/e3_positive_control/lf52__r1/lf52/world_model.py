import numpy as np

import numpy as np

def engine(grid, action, data):
    if action == 6:
        px, py = data['x'], data['y']
        new_grid = grid.copy()
        if 0 <= py < 64 and 0 <= px < 64:
            new_grid[py, px] = 5
        return new_grid
    return grid

def is_level_complete(grid):
    return np.all(grid == 10)

def is_level_complete(grid):
    import numpy as np
    g = np.array(grid)
    if g.shape[0] == 0 or g.shape[1] == 0:
        return False
    # Rule: Win if the bottom row is entirely filled with the target color (green/0)
    # and the grid contains no other colors (only 0 and 1, where 0 is target).
    # Assuming standard ARC color mapping: 0=black, 1=green, 2=red, 3=blue, 4=yellow, 5=magenta, 6=cyan, 7=gray, 8=white, 9=gold.
    # Based on typical ARC-AGI-3 'lf52' patterns, the win state is a full bottom row of the target color.
    # We check if the bottom row is uniform and matches the most frequent color in the grid (excluding background if needed, but here we assume 0 is the target).
    # Simplified rule: Bottom row must be all 0s (black) or all 1s (green) depending on context, but usually 'lf52' implies filling the bottom row with the active color.
    # Let's assume the target is the color that fills the bottom row.
    # General rule: The bottom row is completely filled with a single color, and that color is the one being moved/stacked.
    # Without specific color mapping, we check if the bottom row is uniform.
    # However, usually ARC tasks require a specific color. Let's assume the target is 0 (black) or 1 (green).
    # Given the ambiguity, the most robust general rule for 'lf52' is: Bottom row is full of the target color.
    # Let's assume the target color is 0 (black) based on common ARC 'fill' tasks.
    # Actually, looking at 'lf52' specifically, it often involves filling the bottom row with the color of the falling blocks.
    # Since we don't know the color, we check if the bottom row is uniform and non-empty.
    # But a strict win condition usually requires a specific color.
    # Let's assume the win condition is: Bottom row is all 0s.
    # Wait, 'lf52' is a specific ARC task. It usually involves filling the bottom row with the color 0 (black) or 1 (green).
    # Let's check if the bottom row is all 0s.
    # If the grid is full of 0s, it's a win.
    # Let's refine: The bottom row must be all 0s.
    return np.all(g[-1, :] == 0)
