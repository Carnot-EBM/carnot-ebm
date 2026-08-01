import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a movement-based puzzle where an object (color 2) of size 2x2 or similar moves.
    # Based on the observed transitions, ACTION4 (down), ACTION2 (left/right?), etc.
    # Let's assume standard keyboard directions for ARC games.
    # Action 1: Up, 2: Left, 3: Right, 4: Down.
    # However, the same action number can actually move the object in different directions.
    # la// This is logic based on the<|channel>thought process which I cannot include.
    # 
    # Looking at the deltas:
    # ACTION4: Object color 2 shifts from row 24+ area down and right?
    # ACTION2: Object color 2 shifts left then later shifted again.
    # 
    # In this specific case, we are not given enough information to implement a complex physics engine.
    # But we look at the patterns of changes.
    # Color 2 is the "player" or "object".
    # It looks like it interacts with blocks of color 15 (the target/goal?).
    # 
    # Given the strict rules, I will implement a simple movement model for the player object (color 2).
    # The same action might be interpreted as direction if data is None.
    # 
    # Based on observed transitions:
    # ACTION4 -> moves something.
    # ACTION2 -> moves something.
    # Let's map Action 1: Up, 2: Left, 3: Right, 4: Down.
    # Wait, looking at r21c39:2x1 in ACTION4 delta... that's moving things.
    # 
    # Actually, let's try to find the 'player' (color 2) coordinates first.
    # Then move them based on the action.
    # 
    # Since I cannot reason, I will provide a basic implementation that handles the movement of color 2 cells.

    # Find all cells of color 2
    coords = np.argwhere(grid == 2)
    if len(coords) == 0:
        return grid.copy()

    # Calculate center or top-left of the object
    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    
    # Define directions for actions 1-4
    # 1: Up, 2: Left, 3: Right, 4: Down
    dr, dc = {1: (-1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}.get(action, (0, 0)) # Placeholder
    # Correcting based on observed ACTION4 moving things down/right and ACTION2 moving left.
    # Let's try Action 4: Down, Action 2: Left, Action 3: Right, Action 1: Up.
    dir_map = {1: (-1, 0), 2: (0, -1), 3: (0, 1), 4: (1, 0)}
    move = dir_map.get(action, (0, 0))
    
    new_grid = grid.copy()
    
    # Simple movement: move all color 2 cells by the offset
    for r, c in coords:
        nr, nc = r + move[0], c + move[1]
        if 0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]:
            # Check for collisions or target interaction?
            # In this game, it looks like they overwrite other colors.
            new_grid[nr, nc] = 2
            # The old position becomes background (color 5)
            new_grid[r, c] = 5
            
    return new_grid

def is_level_complete(grid):
    # A level is complete if certain conditions are met.
    # Usually, this means a specific object reaches a goal or blocks of a certain color disappear.
    # Looking at the initial grid, there are large areas of color 15.
    # If those are gone, maybe it's complete.
    return np.sum(grid == 15) == 0