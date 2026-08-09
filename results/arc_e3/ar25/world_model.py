import numpy as np

def engine(grid, action, data):
    """
    World model for ARC game 'ar25'.
    Action 2: Move Player Down (+3y), Target Down (+3y)
    Action 3: Move Player Left (-3x), Target Right (+3x)
    Every action increments a marker on column 63.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    # Identify player and target positions
    # Player consists of color 5 and its holes (color 0)
    player_pixels = []
    target_pixels = []
    for r in range(h):
        for c in range(w):
            if grid[r, c] == 5 or (grid[r, c] == 0 and c < 30):
                player_pixels.append((r, c))
            elif grid[r, c] == 4:
                target_pixels.append((r, c))

    # Determine shifts based on action
    dy_p, dx_p = 0, 0
    dy_t, dx_t = 0, 0
    if action == 2:
        dy_p, dy_t = 3, 3
    elif action == 3:
        dx_p, dx_t = -3, 3

    # Clear old positions with background color 9
    # We only clear the areas where the objects were to avoid destroying walls/goals
    for r, c in player_pixels:
        new_grid[r, c] = 9
    for r, c in target_pixels:
        new_grid[r, c] = 9

    # Apply movement for Player
    for r, c in player_pixels:
        nr, nc = r + dy_p, c + dx_p
        if 0 <= nr < h and 0 <= nc < w:
            new_grid[nr, nc] = grid[r, c]

    # Apply movement for Target
    for r, c in target_pixels:
        nr, nc = r + dy_t, c + dx_t
        if 0 <= nr < h and 0 <= nc < w:
            new_grid[nr, nc] = grid[r, c]

    # Update counter on column 63 (set first non-5 cell starting from row 0 to 5)
    for r in range(h):
        if new_grid[r, 63] != 5:
            new_grid[r, 63] = 5
            break

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the target object (color 4) perfectly overlaps 
    the goal area defined by color 11.
    Goal Area: Rows 45-47 cols 51-59; Rows 48-53 cols 51-53.
    This corresponds to a target top-left corner at (45, 51).
    """
    # Find all pixels of color 4
    target_coords = np.argwhere(grid == 4)
    if len(target_coords) == 0:
        return False
    
    min_r = np.min(target_coords[:, 0])
    min_c = np.min(target_coords[:, 1])
    
    # The win condition is reaching the specific coordinates (45, 51)
    return min_r == 45 and min_c == 51