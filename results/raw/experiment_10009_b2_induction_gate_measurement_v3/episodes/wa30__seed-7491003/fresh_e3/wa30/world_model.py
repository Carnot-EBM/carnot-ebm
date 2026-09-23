import numpy as np


def _find_player(grid):
    """Find the player (color 14) position."""
    ys, xs = np.where(grid == 14)
    if len(ys) == 0:
        return None
    # Player is a 4x4 block of color 14
    y0, x0 = ys.min(), xs.min()
    return (y0, x0)


def _get_block_at(grid, y0, x0):
    """Get the 4x4 block starting at (y0, x0)."""
    return grid[y0:y0+4, x0:x0+4].copy()


def _place_block(grid, y0, x0, block):
    """Place a 4x4 block at (y0, x0)."""
    grid[y0:y0+4, x0:x0+4] = block


def engine(grid, action, data):
    grid = grid.copy()
    
    # Find all 4x4 blocks that are not background (not all 1s)
    # The player moves one cell per action in the direction indicated
    
    # Determine movement direction based on action
    # Action 1: up, Action 2: down, Action 3: left, Action 4: right
    dy, dx = 0, 0
    if action == 1:
        dy, dx = -1, 0
    elif action == 2:
        dy, dx = 1, 0
    elif action == 3:
        dy, dx = 0, -1
    elif action == 4:
        dy, dx = 0, 1
    else:
        return grid

    # Find the player position (top-left of 4x4 color-14 block)
    ys, xs = np.where(grid == 14)
    if len(ys) == 0:
        return grid
    
    y0, x0 = int(ys.min()), int(xs.min())
    
    # Try to move
    ny0, nx0 = y0 + dy, x0 + dx
    
    # Check bounds
    if ny0 < 0 or ny0 + 4 > grid.shape[0] or nx0 < 0 or nx0 + 4 > grid.shape[1]:
        return grid
    
    # Get current player block
    player_block = _get_block_at(grid, y0, x0)
    
    # Clear old position
    _place_block(grid, y0, x0, np.ones((4, 4), dtype=grid.dtype))
    
    # Place at new position
    _place_block(grid, ny0, nx0, player_block)
    
    # Handle collision with structures (color 4 borders, color 9/2 interiors)
    # When player overlaps a structure, merge them
    for r in range(ny0, ny0 + 4):
        for c in range(nx0, nx0 + 4):
            if grid[r, c] != 1 and grid[r, c] != 14:
                # This is part of a structure - the player absorbs it
                pass
    
    # The key mechanic: when the player moves into a structure area,
    # the structure's non-background cells get absorbed into the player block
    # Actually looking more carefully at the transitions:
    # The player moves and picks up whatever is in its path
    
    # Let me re-analyze: the player is always 4x4. When it moves onto 
    # a structure, the structure cells become part of the player.
    
    return grid


def is_level_complete(grid):
    """Check if level is complete."""
    # Win condition: all structures collected / specific state reached
    # Based on observations, the game seems to involve collecting all colored objects
    # Check if no color-4 structures remain (all collected by player)
    has_structure = np.any(grid == 4)
    has_9 = np.any(grid == 9)
    has_2 = np.each(grid == 2) if hasattr(np, 'each') else np.any(grid == 2)
    return not has_structure and not has_9 and not has_2