import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION 4: Right movement of a specific object/block
    # ACTION 5: Left movement or toggle?
    # ACTION 1: Upward movement of a sequence of blocks
    # The game seems to be a puzzle where objects move in response to actions.
    # Let's identify the "player" or moving parts.
    # Based on the same delta patterns, it's a<|channel>thought
    # that there are certain entities (blocks) and that they move.
    #
    # Looking at the deltas for ACTION 4:
    # r11c39:5x1, r11c42:9x1 ... this looks like a vertical bar of color 9 shifting right by 3 pixels.
    # Similarly, ACTION 1 shifts things up.
    #
    # Since we don't have enough information to fully define the physics,
    # but the pattern is very clear:
    # ACTION 4 -> shift some columns of color 9 right.
    # ACTION 1 -> shift some rows of color 11/0/etc. up.
    #
    # However, the most consistent thing across all transitions is the change in the bottom row (r63).
    # This suggests a progress counter or state tracker.
    #
    # Given the constraints and the nature of ARC-AGI, let's implement a simple movement model.
    # The observed actions seem to be directional movements of specific colors.
    # Action 4: Right, Action 5: Left, Action 1: Up, Action 2: Down?
    #
    # Let's refine based on the provided data:
    # ACTION 4: Shifts a column of color 9 from c39->c42, then c42->c45, etc.
    # ACTION 1: Shifts blocks of color 11 upwards.
    #
    # Because this is an "executable world model", I will implement the logic that
    # shifts these specific patterns.

    new_grid = grid.copy()
    if action == 4: # Move Right
        # Find vertical bars of color 9 and move them right by 3
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1] - 3):
                if grid[r, c] == 9:
                    # Check if it's part of a vertical bar (simplified)
                    new_grid[r, c+3] = 9
                    new_grid[r, c] = 5 # Background color
    elif action == 5: # Move Left
        for r in range(grid.shape[0]):
            for c in range(3, grid.shape[1]):
                if grid[r, c] == 9:
                    new_grid[r, c-3] = 9
                    new_grid[r, c] = 5
    elif action == 1: # Move Up
        # Shift blocks of color 11 upwards
        for r in range(1, grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 11:
                    new_grid[r-3, c] = 11
                    new_grid[r, c] = 5
    elif action == 2: # Move Down
        for r in range(grid.shape[0]-3):
            for c in range(grid.shape[1]):
                if grid[r, c] == 11:
                    new_grid[r+3, c] = 11
                    new_grid[r, c] = 5

    # Update the progress counter in the bottom row (r63)
    # The observed deltas show r63c58:1x1 -> r63c57:1x1 -> r63c56:1x1 etc.
    # This is a pointer moving left.
    if action in [1, 4, 5]:
        # Find the '1' in the last row and move it left
        last_row = new_grid[-1, :]
        idx = np.where(last_row == 1)[0]
        if len(idx) > 0:
            curr_pos = idx[0]
            if curr_pos > 0:
                new_grid[-1, curr_pos - 1] = 1
                new_grid[-1, curr_pos] = 15 # Background color of last row

    return new_grid

def is_level_complete(grid):
    # Level complete when the pointer in the last row reaches a certain position
    # or all objects are aligned. Based on the data, we don't have a win state grid.
    # Usually, this happens when the counter hits 0 or a target.
    last_row = grid[-1, :]
    idx = np.where(last_row == 1)[0]
    if len(idx) > 0:
        return idx[0] == 0
    return False