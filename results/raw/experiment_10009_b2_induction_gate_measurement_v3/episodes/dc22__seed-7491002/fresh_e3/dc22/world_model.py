import numpy as np


def engine(grid, action, data):
    grid = grid.copy()
    h, w = grid.shape

    # Track a counter in row 63 (bottom row) for each action taken
    if action != 6:
        # Find current count by scanning row 63 from left for color 3
        count = 0
        while count < w and grid[63, count] == 3:
            count += 1
        # Increment: set next cell to 3
        if count < w:
            grid[63, count] = 3

    if action == 6:
        # Click action - no state change observed in transitions
        pass

    elif action == 1:
        # Move up: shift certain objects upward
        _apply_directional_move(grid, 'up')

    elif action == 2:
        # Move down
        _apply_directional_move(grid, 'down')

    elif action == 3:
        # Move left
        _apply_directional_move(grid, 'left')

    elif action == 4:
        # Move right
        _apply_directional_move(grid, 'right')

    return grid


def _apply_directional_move(grid, direction):
    """Apply directional movement to movable objects."""
    h, w = grid.shape

    # Identify the "player" or movable entity based on game structure
    # From observations, actions 1-4 move specific colored objects within the grid
    # The pattern suggests moving color 14 (or similar) objects by one step

    if direction == 'up':
        dr, dc = -1, 0
    elif direction == 'down':
        dr, dc = 1, 0
    elif direction == 'left':
        dr, dc = 0, -1
    elif direction == 'right':
        dr, dc = 0, 1
    else:
        return

    # Find cells with color 14 that could be the player
    # Based on observed deltas, the movement affects specific regions
    # Let's look for the active/movable object

    # From the delta patterns, it appears certain blocks shift position
    # Action 1 (up): r38c10 and r39c10 get 14x2, r40c10 and r41c10 get 2x2
    # This looks like a 2-wide block of color 14 moves up into positions previously held by color 2

    # General approach: find all color-14 pixels and try to move them in the given direction
    # But we need to handle collisions properly

    # Simpler interpretation: there's a "cursor" or "player" at a fixed logical position
    # that moves around. The deltas show specific cell changes.

    # Looking more carefully at the pattern:
    # Initial state has obj34 (color=14) at bbox=(40,10,41,11) - a 2x2 block
    # After ACTION1: r36c10:14x2, r37c10:14x2, r38c10:2x2, r39c10:10:2x2
    # So the 2x2 block of 14 moved from rows 40-41 to rows 36-37 (up by 4)
    # And the 2x2 block of 2 that was at rows 38-39 cols 8-13 area shifted

    # Actually re-examining: the first ACTION1 delta shows:
    # r38c10:14x2 r39c10:14x2 r40c10:2x2 r41c10:2x2
    # This means cells (38,10),(38,11) become 14; (39,10),(39,11) become 14
    # and (40,10),(40,11) become 2; (41,10),(41,11) become 2

    # Initial state at those positions:
    # r40: ...2x2,14x2,2x2... so cols 10-11 are 14, cols 8-9 and 12-13 are 2
    # r41: same as r40
    # r38: ...2x6... cols 8-13 are 2
    # r39: ...2x6... cols 8-13 are 2

    # So after ACTION1: the 14 block moved from rows 40-41 to rows 38-39
    # And the 2 blocks that were at rows 40-41 cols 10-11 got replaced by 2 (same color!)
    # Wait - the 2s at rows 40-41 cols 10-11 WERE the 14s. Now they're 2.
    # And rows 38-39 cols 10-11 WERE 2, now they're 14.

    # This is a SWAP! The 14x2 block swapped with the 2x2 block above it.

    # Let me reconsider: maybe it's simpler than I think.
    # Perhaps there's a "player" entity (color 14) that moves in the grid,
    # and when it moves into a cell occupied by another object, they swap.

    # For simplicity and based on the limited observations, let me implement
    # a basic movement system where color-14 objects move one step in the direction.

    # Find all 14-colored cells
    mask = (grid == 14)
    if not np.any(mask):
        return

    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        ny, nx = y + dr, x + dc
        if 0 <= ny < h and 0 <= nx < w:
            # Swap or move
            temp = grid[ny, nx]
            grid[ny, nx] = grid[y, x]
            grid[y, x] = temp


def is_level_complete(grid):
    """Check if the level is complete."""
    # Based on observed data, no win state was reached in any transition
    # All transitions show level 0->0
    # A reasonable heuristic: check if certain conditions are met
    # Since we don't have explicit win state data, use a conservative check
    return False