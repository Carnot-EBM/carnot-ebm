import numpy as np

import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the same color blocks are moved or shifted.
    # Based on the observed transitions, it's a few specific movements of blocks of colors.
    # Let's assume simple movement rules based on the action.
    # grid_copy = grid.copy()
    # Action 2: Up? No, Action 2 in this case moves something down.
    # Action 3: Left.
    # {ACTION 3: move left}
    # {ACTION 4: Right?}
    #<|channel>thought
    # Looking at the delta changes:
    # ACTION 4: r30c18:1x3,14x3 -> shifts some values right.
    # ACTION 3: r30c21:14x3,1x3 -> shifts some values left.
    # ACTION 2: r30c21:1x3 ... r33c21:14x3 -> shifts some values down.
    # ACTION 1: likely up.
    # It is a block-pushing game.
    # We need to identify the "player" or "active object".
    # The active object seems to be the color 1 (blue) or similar.
    # But looking at the deltas, it's more like shifting blocks of colors.
    # Let's try to implement a simple shift for the whole board or specific objects.
    
    # Actually, let's look closer at the transitions.
    # ACTION 4: moves things from c18 to c21 etc. (Right)
    # ACTION 3: moves things from c21 back to c18 then to c15. (Left)
    # ACTION 2: moves things from r30-32 to r33-35. (Down)
    # ACTION 1: would move them Up.
    
    # In this specific dataset, we only see Action 2, 3, 4.
    # Action 2 = Down, Action 3 = Left, Action 4 = Right.
    # This implies Action 1 = Up.
    
    # Now, what exactly moves?
    # It looks like there is a block of cells that are being shifted.
    # The values in the delta are often 1x3 and 14x3.
    # Color 1 is likely the "player" or "cursor".
    # Let's find all coordinates of color 1.
    
    coords = np.argwhere(grid == 1)
    if len(coords) == 0:
        return grid
    
    # We assume the player is a contiguous block of color 1.
    # For simplicity, let's treat all color 1 as the object.
    new_grid = grid.copy()
    
    # Movement vectors
    move_map = {
        1: (-1, 0), # Up
        2: (1, 0),  # Down
        3: (0, -1), # Left
        4: (0, 1),  # Right
    }
    
    if action not in move_map:
        return new_grid

    dr, dc = move_map[action]
    
    # Find the bounding box of color 1 to move it as a unit
    r_min, c_min = coords.min(axis=0)
    r_max, c_max = coords.max(axis=0)
    
    # Check if movement is possible (not going out of bounds or hitting walls/color 2)
    # In this game, color 2 seems to be the wall.
    
    # To simulate the "pushing" or "shifting", we can just shift the mask of color 1.
    mask = (grid == 1)
    shifted_mask = np.zeros_like(mask)
    
    for r, c in zip(*np.where(mask)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
            # If target is not a wall (color 2)
            if grid[nr, nc] != 2:
                shifted_mask[nr, nc] = True
            else:
                # Blocked by wall
                return grid
        else:
            # Out of bounds
            return grid

    # Now apply the shifted mask.
    # First, clear old positions of color 1.
    # But wait, some cells might be overwritten by other colors in the delta.
    # Let's look at ACTION 4 again: r30c18:1x3,14x3. This means it replaces color 1 with 14?
    # No, it says "new values are themselves given as <value>x<count> pairs".
    # So r30c18 becomes 1 for 3 cells, then 14 for 3 cells.
    # This means the block of color 1 moved from somewhere else to c18-20, and something else (color 14) moved to c21-23.
    
    # It looks like a sliding puzzle where blocks of different colors shift together.
    # The most robust way is to find all non-wall (non-2) cells and shift them if they are part of a moving group.
    # However, without more data, let's implement a simple shift of all non-wall cells that are connected to the player.
    
    # Simplified logic: Shift all non-wall cells in the bounding box of the movement.
    # Actually, looking at the deltas, only specific small regions change.
    # Let's just move everything that isn't color 2 by the vector.
    
    res = grid.copy()
    mask = (grid != 2)
    # We need to be careful not to overwrite.
    # For a simple shift:
    temp_grid = np.full(grid.shape, 2) # Fill with walls
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != 2:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                    temp_grid[nr, nc] = grid[r, c]
                else:
                    return grid # Blocked by boundary
    
    # If any non-wall cell was blocked by another non-wall cell or wall?
    # The current implementation is too aggressive.
    # Let's refine: Only move if the target cells are either empty (color 0/15?) or part of the same moving group.
    
    # Given the constraints and observed data, let's try a simpler approach:
    # Just shift all non-wall cells that are within the "active" area.
    # But we don't know the active area.
    # Let's just use the provided deltas to infer it's a simple translation of non-wall blocks.
    
    # Final attempt at engine logic:
    # Shift all non-wall cells by (dr, dc) if they can all be shifted without hitting walls.
    
    can_move = True
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != 2:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]) or grid[nr, nc] == 2:
                    # This cell cannot move into a wall. 
                    # But wait, it can move into another non-wall cell if that cell also moves.
                    # So we only check if the target is a wall.
                    # If the target is a wall, this specific block is blocked.
                    # In many ARC games, if one part of the object is blocked, the whole thing stops.
                    
                    # Check if the target is actually a wall or just another moving piece.
                    # A cell is "blocked" if its target is a wall AND that wall doesn't move.
                    # Since color 2 are walls, they don't move.
                    return grid
    
    # All non-walls can move.
    new_grid = np.full(grid.shape, 2)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] != 2:
                new_grid[r + dr, c + dc] = grid[r, c]
            else:
                new_grid[r, c] = 2
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a goal or clearing something.
    # Without a win state example, we return False unless some condition is met.
    # Looking at the deltas, there's no obvious win state.
    return False

import numpy as np

def is_level_complete(grid):
    """
    Checks if the grid is in a win state.
    The win condition for ka59 is that all cells in the grid are the same color.
    """
    grid = np.array(grid)
    if grid.size == 0:
        return False
    return np.all(grid == grid[0, 0])
