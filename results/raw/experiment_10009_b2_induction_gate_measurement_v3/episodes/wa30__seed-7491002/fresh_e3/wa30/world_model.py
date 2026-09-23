import numpy as np


def _find_player(grid):
    """Find the player (color 14) position."""
    positions = np.argwhere(grid == 14)
    if len(positions) == 0:
        return None
    # Player is a 4x4 block of color 14
    for r in range(len(positions)):
        row, col = positions[r]
        if (row + 3 < grid.shape[0] and col + 3 < grid.shape[1]
                and grid[row:row+4, col:col+4].all() == 14):
            return (int(row), int(col))
    return None


def _is_solid(grid, r, c):
    """Check if cell is solid (not background 1 or empty 0)."""
    if r < 0 or r >= grid.shape[0] or c < 0 or c >= grid.shape[1]:
        return True
    v = grid[r, c]
    return v not in (0, 1)


def _can_move(grid, pr, pc, dr, dc):
    """Check if player can move by (dr, dc)."""
    nr, nc = pr + dr, pc + dc
    # Check all 4 corners of the 4x4 player block at new position
    for i in range(4):
        for j in range(4):
            if _is_solid(grid, nr + i, nc + j):
                return False
    return True


def engine(grid, action, data):
    grid = grid.copy()
    
    # Find player
    pos = _find_player(grid)
    if pos is None:
        return grid
    
    pr, pc = pos
    
    # Determine movement direction based on action
    # Action 1 = up, 2 = down, 3 = left, 4 = right (inferred from transitions)
    directions = {
        1: (-1, 0),   # up
        2: (1, 0),    # down
        3: (0, -1),   # left
        4: (0, 1),    # right
    }
    
    if action not in directions:
        return grid
    
    dr, dc = directions[action]
    
    # Try to move step by step (player can slide multiple cells)
    moved = False
    while _can_move(grid, pr, pc, dr, dc):
        # Clear old position
        grid[pr:pr+4, pc:pc+4] = 1
        # Place new position
        nr, nc = pr + dr, pc + dc
        grid[nr:nr+4, nc:nc+4] = 14
        pr, pc = nr, nc
        moved = True
    
    # If player moved and hit a wall at the bottom row area, check for counter
    # The bottom row (63) has color 7 which acts as a counter/timer
    if moved:
        # Check if player is near or on the bottom boundary
        # From transitions, when player moves, sometimes r63cXX changes from 7 to 4
        pass

    return grid


def is_level_complete(grid):
    """Check if level is complete - all 9-colored areas are cleared."""
    # Win condition: no more color 9 pixels remaining in the grid
    # (all target areas have been collected/cleared)
    return not np.any(grid == 9)