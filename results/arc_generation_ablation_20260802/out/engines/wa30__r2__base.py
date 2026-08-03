import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this seems to be a puzzle game where objects move or change state.
    # The provided same-level transitions are not even changing the level index.
    # The same-level transitions show changes in specific regions of thes grid.
    # Action 4 corresponds to some movement/interaction.
    # Action 1 corresponds to some movement/or interaction.
    # Action <|channel>thought
    # Since the actual logic is very complex and the limited observations are only a few steps,
    # we'll implement a simple rule based on the delta patterns seen.
    
    new_grid = grid.copy()
    if action == 4:
        # This looks like it might be moving an object at (32, 32) to (32, 36), etc.
        # Find current position of color 0 (the "player" or "cursor")
        # Note: In the initial grid, color 0 is at r32c35, r33c35, r34c35, r35c35.
        # It moves by 4 columns each time ACTION 4 is called.
        pos = np.where(grid == 0)
        if pos[0].size > 0:
            y_min, x_min = pos[0][0], pos[1][0]
            # Move the block of zeros
            # We assume the block size is 4x1
            for i in range(4):
                new_grid[y_min+i, x_min] = 1
                new_grid[y_min+i, x_min + 4] = 0 if (x_min + 4 < grid.shape[1]) else 0
            # But wait, the delta says r32c32:1x4, 14x3, 0x1... this means it's replacing a span.
            # Let's try to shift the zero-block right by 4 units.
            # The original position was c35. Delta 1: r32c32:1x4, 14x3, 0x1 -> new col 35 is 0? No.
            # Actually, let's look at the deltas again.
            # ACTION 4: r32c32:1x4, 14x3, 0x1. This replaces cells from c32 to c38.
            # Original: r32c32=1, c33=1, c34=1, c35=0, c36=14, c37=14, c38=14.
            # New: c32=1, c33=1, c34=1, c35=1, c36=14, c37=14, c38=14, c39=0? No.
            # Let's simplify: Action 4 moves the '0' block right by 4 columns.
            pass

    # Given the extreme sparsity and complexity of the provided transitions,
    # we will implement a basic movement model for the "player" (color 0).
    # Find the player position (the column where color 0 exists)
    player_pos = np.where(grid == 0)
    if player_pos[0].size > 0:
        py, px = player_pos[0][0], player_pos[1][0]
        # Block size is 4x1
        for i in range(4):
            new_grid[py+i, px] = 1
        
        if action == 4: # Right
            nx, ny = px + 4, py
        elif action == 1: # Up
            nx, ny = px, py - 4
        elif action == 3: # Down
            nx, ny = px, py + 4
        elif action == 5: # Left
            nx, ny = px - 4, py
        else:
            return new_grid
        
        if 0 <= nx < grid.shape[1] and 0 <= ny < grid.shape[0]:
            for i in range(4):
                if ny+i < grid.shape[0]:
                    new_grid[ny+i, nx] = 0
    
    return new_grid

def is_level_complete(grid):
    # The win state is not provided, but usually it's when the player reaches a target.
    # In this game, color 4 seems to be walls/targets.
    # Check if any part of the player (color 0) is overlapping with a specific area or color.
    # Let's assume completion happens when the player block moves into a certain region.
    # Based on ACTION 4 moving towards c63, maybe that's the goal.
    player_pos = np.where(grid == 0)
    if player_pos[0].size > 0:
        px = player_pos[1][0]
        if px >= 55:
            return True
    return False