import numpy as np


def engine(grid, action, data):
    g = grid.copy()
    h, w = g.shape

    # Find the player (color 14) - it's a 4x4 block
    player_pos = None
    for r in range(h - 3):
        for c in range(w - 3):
            if all(g[r + dr][c + dc] == 14 for dr in range(4) for dc in range(4)):
                player_pos = (r, c)
                break
        if player_pos:
            break

    if player_pos is None:
        return g

    pr, pc = player_pos

    # Determine movement direction based on action
    # Action 1 = up, 2 = down, 3 = left, 4 = right (inferred from transitions)
    moves = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if action not in moves:
        return g

    dr, dc = moves[action]

    # The player occupies rows [pr, pr+3], cols [pc, pc+3]
    # It moves one cell at a time. Check if the destination is blocked.
    # A cell is blocked if it contains color 4 (wall/frame) or color 7 (floor boundary).
    
    new_pr = pr + dr
    new_pc = pc + dc

    # Check bounds
    if new_pr < 0 or new_pr + 3 >= h or new_pc < 0 or new_pc + 3 >= w:
        return g

    # Check if any cell in the destination area is a wall (color 4) or floor (color 7)
    for r in range(new_pr, new_pr + 4):
        for c in range(new_pc, new_pc + 4):
            if g[r][c] == 4 or g[r][c] == 7:
                return g

    # Move the player: clear old position, set new position
    # Old position becomes background (color 1)
    for r in range(pr, pr + 4):
        for c in range(pc, pc + 4):
            g[r][c] = 1

    # New position gets color 14
    for r in range(new_pr, new_pr + 4):
        for c in range(new_pc, new_pc + 4):
            g[r][c] = 14

    return g


def is_level_complete(grid):
    # Win condition: all color-9 objects collected / no more color 9 present
    # Based on observation, when the player reaches certain positions, level completes
    # The simplest general rule: check if there are no more collectible items (color 9)
    # Actually from the transitions, it seems like the level doesn't complete during observed frames.
    # Let's check: win state might be when player is at a specific location or all 9s are gone.
    # Since we don't have an explicit win grid, use: no color 9 cells remain (all collected)
    return not np.any(grid == 9)