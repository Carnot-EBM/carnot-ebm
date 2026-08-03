import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The observed transitions suggest a movement of an object/entity represented by color 2 or 15.
    # Action 4 is likely 'down' and Action 2 is 'up'.
    # Based on the same delta patterns, it seems like a player character (color 2)
    # moves in a direction and pushes objects (color 15) or interacts with the<|channel>thought
    # 15s are "blocks" that can be pushed.
    # 15s are "// own blocks" and 2s are "player".
    # 
    # Let's refine the rules based on the provided deltas.
    # ACTION 4: Down
    # ACTION 2: Up
    # ACTION 6: Click (not present in observations)
    # 
    # Looking at the deltas:
    # ACTION 4 (Down): Player (2) moves down, pushing 15s.
    # ACTION 2 (Up): Player (2) moves up, pushing 15s.
    # ACTION 6: Not seen.
    # 
    # Actually, looking closer at the deltas for ACTION 4:
    # r21c39:2x1 -> color 2 appears at (21, 39).
    # The movement pattern of color 2 is clearly shifting.
    # 
    # Since we don't have enough data to fully simulate the physics, 
    # but the prompt asks for a simple general rule, let' same as follows:
    # Color 2 is the player. Color 15 is a pushable block.
    # Color 0 is empty/void. Color 5 is background.
    # Color 4 is wall.
    # 
    # Let's implement a basic grid-based movement and push logic.

    new_grid = grid.copy()
    
    # Find player position
    player_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 2:
                player_pos = (r, c)
                break
        if player_pos: break
    
    if not player_pos:
        return new_grid

    # Action mapping based on common ARC patterns:
    # ACTION 1: Up, 2: Down, 3: Left, 4: Right? No, observed ACTION 4 moves "down"ish and ACTION 2 moves "up"ish.
    # Wait, looking at the deltas again:
    # ACTION 4: Player starts around r21c39, ends up moving down to r37c43.
    # ACTION 2: Player starts around r37c43, ends up moving back up or shifting.
    # Actually, let's look at the coordinates of color 2 in the deltas.
    # ACTION 4 delta: r21c39:2x1 ... r36c43:2x2, r37c43:2x1.
    # ACTION 2 delta: r32c38:2x9... r45c38:2x9.
    # This is confusing. Let's try a simpler approach.
    # The game seems to be about pushing blocks (15) with a player (2).
    
    dr, dc = 0, 0
    if action == 1: dr, dc = -1, 0 # Up
    elif action == 2: dr, dc = 1, 0  # Down
    elif action == 3: dr, dc = 0, -1 # Left
    elif action == 4: dr, dc = 0, 1  # Right
    elif action == 5: dr, dc = 0, 0   # None
    
    # Correcting based on observed transitions:
    # Transition 1: Action 4 -> Player moves from ~r21 to ~r37. That's DOWN.
    # Transition 2: Action 2 -> Player moves from ~r37 back up? No, it says r45c38:2x9. That's further DOWN.
    # Wait, the deltas are "changed cells". If color 2 was at (37, 43) and now it's at (45, 38), it moved down-left.
    
    # Let's use standard directions for a generic world model if not clear:
    # 1:Up, 2:Down, 3:Left, 4:Right.
    if action == 1: dr, dc = -1, 0
    elif action == 2: dr, dc = 1, 0
    elif action == 3: dr, dc = 0, -1
    elif action == 4: dr, dc = 0, 1
    else: return new_grid

    r, c = player_pos
    nr, nc = r + dr, c + dc
    
    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
        if grid[nr, nc] == 5 or grid[nr, nc] == 0: # Move into empty/bg
            new_grid[r, c] = 5
            new_grid[nr, nc] = 2
        elif grid[nr, nc] == 15: # Push block
            nnr, nnc = nr + dr, nc + dc
            if 0 <= nnr < grid.shape[0] and 0 <= nnc < grid.shape[1]:
                if grid[nnr, nnc] == 5 or grid[nnr, nnc] == 0:
                    new_grid[nnr, nnc] = 15
                    new_grid[nr, nc] = 2
                    new_grid[r, c] = 5
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves blocks (15) being on specific targets (maybe color 0?)
    # Or just a general condition. Since no win state was provided, we check for common patterns.
    # In many ARC games, the goal is to clear all blocks of a certain type or move them to a target.
    # Let's assume it's complete if there are no more blocks of color 15.
    return not np.any(grid == 15)