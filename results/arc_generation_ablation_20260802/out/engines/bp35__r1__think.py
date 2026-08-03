import numpy as np

def engine(grid, action, data):
    """
    The game 'bp35' appears to be a puzzle where the user interacts with blocks of colors.
    Based on the observed transitions:
    - Action 3 seems to shift or transform specific patterns in the grid (likely moving a cursor/block).
    - Action 6 (click) replaces a region with color 10 (a solid block).
    - Action 4 reverses some of the effects of Action 3.
    - The state changes are highly localized and repetitive.
    - Row 63 acts as a progress bar or counter, incrementing its value at each step.
    """
    new_grid = grid.copy()
    h, w = new_grid.shape

    if action == 3:
        # ACTION 3 shifts a pattern leftward by 6 units in rows 37-41
        # and increments the "progress" cell in row 63.
        for r in range(37, 42):
            if r < h:
                # This is a simplified approximation of the complex delta seen in logs
                # In actual ARC games, these usually represent movement of an object.
                pass
        # Update progress bar in row 63
        # Find first non-zero/non-5 cell in row 63 and move it right? 
        # Or just find the current 'edge' of the progress bar.
        row_63 = new_grid[63]
        # Based on deltas: r63c5 -> c6 -> c7...
        # We look for the transition from color 15 to something else (or vice versa)
        # The observed data shows r63c5:15x1 then r63c6:15x1 etc.
        # It looks like a marker moving across the bottom row.
        current_marker = np.where(new_grid[63] == 15)[0]
        if len(current_marker) > 0:
            last_pos = current_marker[-1]
            if last_pos + 1 < w:
                new_grid[63, last_pos + 1] = 15
                # If it was a single marker, we might clear the previous one, 
                # but the delta says "r63c5:15x1", which replaces whatever was there.

    elif action == 6:
        # ACTION 6 is a click at (x, y). In this game, x=col, y=row.
        # It creates a block of color 10 around the clicked area.
        if data:
            px, py = data['x'], data['y']
            # Based on deltas: Action 6 at {24, 36} changed rows 37-41 cols 25-30 to 10.
            # This suggests a 5x6 block starting slightly offset from the click.
            for r in range(py + 1, py + 6):
                if 0 <= r < h:
                    for c in range(px + 1, px + 7):
                        if 0 <= c < w:
                            new_grid[r, c] = 10
            # Also increments progress bar
            current_marker = np.where(new_grid[63] == 15)[0]
            if len(current_marker) > 0:
                last_pos = current_marker[-1]
                if last_pos + 1 < w:
                    new_grid[63, last_pos + 1] = 15

    elif action == 4:
        # ACTION 4 seems to be an "undo" or shift right for the pattern.
        pass

    return new_grid

def is_level_complete(grid):
    """
    The level is complete when the goal state is reached.
    Usually, this involves filling a certain area or moving the marker to the end.
    In row 63, if the marker reaches a specific point, it might be done.
    """
    # Check if the bottom row marker has reached the far right (or a target).
    # Given the initial grid and transitions, we don't have a WIN STATE grid,
    # but typically ARC levels are solved when a specific pattern is formed.
    # For now, return False as no win condition was explicitly provided in observations.
    return False