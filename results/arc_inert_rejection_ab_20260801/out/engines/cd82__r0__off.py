import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # This game appears to be a puzzle where objects move or change state based on actions.
    # Based on the observed transitions, ACTION 2 and 4 are directional movements.
    # Action 2: Up? No, looking at the row indices in deltas, it's more like Down/Right.
    # Action 4: Right?
    # Let's analyze the movement of color 2 (the agent/player) and the blocks.
    # Color 2 is likely the player.
    # Color 15 is likely the "trail" or "collectible".
    # Color 0 is "wall/obstacle".
    # Color 3, 4, 5 are background/static elements.
    # The observations show that when ACTION 4 is taken, things shift rightward.
    # Find the player position.
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 2:
                player_pos = (r, c)
                break
        if player_pos: break

    if player_pos is None:
        return grid.copy()

    pr, pc = player_pos
    dr, dc = 0, 0
    if action == 1: dr, dc = -1, 0 # Up
    elif action == 2: dr, dc = 0, 1  # Right
    elif action == 3: dr, dc = 0, -1 # Left
    elif action == 4: dr, dc = 1, 0  # Down
    elif action == 6:
        # Click action; usually not a specific movement but interaction.
        # In this game's context, it's likely just clicking on something.
        #<|channel>thought
        # No matter what the data provided, we can
        # reconstruct the same state change based on the observed deltas.
        pass

    # Simple move logic for the agent (color 2).
    # Color 5 is empty space/background.
    # Color 0 is wall.
    # Color 15 is "collectible" or "trail".
    # The observations show that when ACTION 4 is taken, row indices increase (Down),
    # and column indices increase (Right). This suggests Action 4 might be 'down-right'.
    # Let's refine the directions based on the observed transitions:
    # ACTION 4 delta: r21c39...r37c43. Row increases. Column increases.
    # ACTION 2 delta: r21c39...r45c38. Row increases. Column changes.
    # ACTION 2 again: r32c38...r56c39. Row increases further.
    # It looks like gravity or a sliding mechanism.
    
    # Since I cannot fully deduce the complex physics from such limited data,
    # I will implement a basic movement model where color 2 moves in a direction
    # and pushes blocks of other colors if they are not walls (0).
    
    new_grid = grid.copy()
    if action == 1: dr, dc = -1, 0
    elif action == 2: dr, dc = 0, 1
    elif action == 3: dr, dc = 0, -1
    elif action == 4: dr, dc = 1, 0
    else: return new_grid

    nr, nc = pr + dr, pc + dc
    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
        if grid[nr, nc] != 0: # Not a wall
            # If moving into a collectible (15), it might be consumed or changed to background (5)
            if grid[nr, nc] == 15:
                new_grid[nr, nc] = 5
            # Move player
            new_grid[pr, pc] = 5
            new_grid[nr, nc] = 2
    return new_grid

def is_level_complete(grid):
    # Level complete usually happens when all collectibles (15) are gone or agent reaches a goal.
    # In the observed data, color 15 exists in several places.
    # The win state is not provided, but we can assume completion if no more 15s remain.
    return np.sum(grid == 15) == 0