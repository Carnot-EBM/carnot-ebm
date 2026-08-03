import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Based on observed transitions, this looks like a puzzle where certain blocks move in response to actions.
    # Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right, Action 5: ?, Action 6: Click, Action 7: ?
    # The changes are own-block movements of specific color patterns.
    # Find the current position of the "player" or "active object" which is represented by the same pattern.
    # In the observed transitions, ACTION1 moves things UP, ACTION3 moves things LEFT, ACTION4 moves things RIGHT.
    # la//C = logical resolution.
    # Logical resolution is 64x64.
    # Grid contains colors [0, 1, 3, 4, 5, 8, 9, 11, 12].
    # Color 4 is background/wall.
    # Color 5 is a border/obstacle.
    # Color 3 is a path/floor.
    # Colors 9 and 12 are moving parts.
    # Let's identify the active block (the one that moves).
    # The active block seems to be composed of cells with values other than 4, 5, 3.
    # These blocks move as a unit.
    #<|channel>thought
    # The movement is constrained by walls (color 4) and obstacles (color 5).
    # Action 1: Up (-y), Action 2: Down (+y), Action 3: Left (-x), Action 4: Right (+x).
    # Looking at the deltas, we see specific patterns of color 9 and 12 shifting.
    # It' same-block shift.
    # In ACTION1, r40c19 becomes 12x5, etc. This means the block shifted up.
    # We need to find all "moving" cells (not 4, 5, 3) and shift them if possible.
    # However, it looks like only ONE group of connected components shifts.
    # Let's assume any cell not in {4, 5, 3} is part of the moving entity.
    # Any cell not in {4, 5, 3} will move together.
    # But wait, there are multiple such groups? No, let's check.
    # Initial grid has several areas with colors other than 4, 5, 3.
    # For example, rows 8-16 have a block, rows 25-49 have blocks, rows 53-62 have blocks.
    # The transitions show that ONLY one specific block moves per action.
    # Which one? The one that can actually move in that direction without hitting walls/obstacles.
    # Actually, looking at the deltas:
    # Action 1 (Up): Cells in r40-r49 moved to r35-r44.
    # Action 3 (Left): Cells in r45-r49 shifted from c24-c28 to c19-c23.
    # This suggests we shift all "movable" cells if they don't collide.
    # Movable cells = those with values NOT in {4, 5, 3}.
    # Let's refine: movable cells are those that aren't part of the static environment.
    # Static environment = {4, 5, 3} AND maybe some others.
    # Looking at the initial grid, color 11 is also present (r61, r62). It doesn't seem to move.
    # So moving colors are {0, 1, 8, 9, 12}.
    # Wait, color 0 is used in r31, r32. Color 1 is used in r32, r33.
    # These are inside the path (color 3). They might be goals or holes.
    # The blocks are made of colors {9, 12}.
    # Let's try shifting all connected components of {9, 12} if they can move.

    moving_colors = {9, 12}
    
    # Find all coordinates of moving blocks
    coords = np.argwhere(np.isin(grid, list(moving_colors)))
    if coords.size == 0:
        return grid.copy()
    
    # Direction vectors for actions 1-4
    dirs = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}
    if action not in dirs:
        return grid.copy()
    
    dy, dx = dirs[action]
    
    # We need to identify which block moves. In this game, it seems only one "active" block moves.
    # But we don't know which one. Let's assume any block that *can* move in that direction does.
    # A block can move if all its cells shifted by (dy, dx) land on a cell that is NOT a wall/obstacle.
    # Walls/Obstacles are colors {4, 5}. Color 3 is floor.
    # Also, the block cannot move into another moving block.
    
    # To simplify, let's find connected components of moving colors.
    from scipy.ndimage import label
    labeled_array, num_features = label(np.isin(grid, list(moving_colors)))
    
    new_grid = grid.copy()
    for i in range(1, num_features + 1):
        block_coords = np.argwhere(labeled_array == i)
        
        # Check for collision
        can_move = True
        for r, c in block_coords:
            nr, nc = r + dy, c + dx
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                can_move = False
                break
            if grid[nr, nc] in {4, 5}:
                can_move = False
                break
            # Collision with other blocks?
            if labeled_array[nr, nc] != 0 and labeled_array[nr, nc] != i:
                can_move = False
                break
        
        if can_move:
            # Move the block: clear old positions, set new ones
            for r, c in block_coords:
                new_grid[r, c] = grid[r, c] if own_is_still_there(new_grid, r, c, block_coords, dy, dx) else 3 # This is tricky
                # Let's just use a temporary grid
                pass

    # Correct way to move connected components:
    final_grid = grid.copy()
    # To handle "floor" replacement, we need to know what was there.
    # Assume floor is color 3.
    
    # Re-do movement logic
    temp_grid = grid.copy()
    for i in range(1, num_features + 1):
        block_coords = np.argwhere(labeled_array == i)
        dy, dx = dirs[action]
        can_move = True
        for r, c in block_coords:
            nr, nc = r + dy, c + dx
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or \
               grid[nr, nc] in {4, 5} or \
               (labeled_array[nr, nc] != 0 and labeled_array[nr, nc] != i):
                can_move = False
                break
        
        if can_move:
            # Move this specific block
            for r, c in block_coords:
                # The cell it leaves becomes the background/floor if no other part of same block moves into it
                # But since it's a translation, that's always true unless it's a complex shape.
                # For simple translations, we just need to be careful.
                pass

    # Let's use a simpler approach: shift all moving colors together if they don't collide with walls.
    # This matches the observed deltas where multiple rows shifted.
    
    mask = np.isin(grid, list(moving_colors))
    shifted_mask = np.zeros_like(mask)
    
    # Shift mask
    if action == 1: # Up
        shifted_mask[0:-1, :] = mask[1:, :]
    elif action == 2: # Down
        shifted_mask[1:, :] = mask[0:-1, :]
    elif action == 3: # Left
        shifted_mask[:, 0:-1] = mask[:, 1:]
    elif action == 4: # Right
        shifted_mask[:, 1:] = mask[:, 0:-1]
    
    # Collision check: can't move into {4, 5}
    collision = np.zeros_like(mask, dtype=bool)
    if action == 1: collision[1:, :] = np.isin(grid[1:, :], {4, 5}) & mask[1:, :] # Wait, this is wrong
    # Correct collision: a cell moves to (nr, nc). If grid[nr, nc] is wall, it's a collision.
    
    # Let's just implement the shift and then revert if any cell hit a wall.
    res_grid = grid.copy()
    coords = np.argwhere(mask)
    new_coords = coords + [dirs[action]]
    
    possible = True
    for nr, nc in new_coords:
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] in {4, 5}:
            possible = False
            break
            
    if possible:
        # Move all moving colors
        # First, set all current moving color positions to floor (3)
        # But only those that aren't being moved into.
        # Since it's a translation, we can just do:
        temp_moving = grid[mask]
        res_grid[mask] = 3
        for i in range(len(coords)):
            nr, nc = new_coords[i]
            res_grid[nr, nc] = temp_moving[i]
        return res_grid

    return grid.copy()

def own_is_still_there(grid, r, c, block_coords, dy, dx):
    for br, bc in block_coords:
        if br + dy == r and bc + dx == c:
            return True
    return False

def is_level_complete(grid):
    # Win state usually involves blocks reaching specific targets (color 0, 1).
    # Or simply the absence of certain conditions.
    # Given no win state provided, let's assume it's complete if any moving block covers a target cell.
    # Target cells are {0, 1}.
    moving_colors = {9, 12}
    targets = {0, 1}
    # If any cell that was a target is now a moving color
    # But we don't have the initial grid to compare.
    # Let's check if there are any target cells left uncovered?
    # No, let's just return False as we don't have enough info for the win condition.
    return False