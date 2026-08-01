import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # Action mapping based on observed transitions:
    # ACTION1: Up, ACTION3: Right, ACTION4: Down/Left? No, let's look at the deltas.
    # ACTION4 seems to move a specific object (color 14) or shift something.
    # ACTION1 moves an object up.
    # ACTION3 moves an object right.
    # ACTION5 might be "undo" or clear.
    
    # Let's refine movement logic.
    # The game involves moving blocks of colors.
    # Color 14 is a player-like object.
    # Color 0 is a hole/target.
    # Color 7 is a floor/boundary.
    # Color 4 is a wall/obstacle.
    # Color 9 is a key/collectible.
    # Color 2 is a special block.
    
    # Based on the observations:
    # ACTION4 shifts color 14 block horizontally (it starts at r32c32:r35c34).
    # ACTION1 moves it vertically (up).
    # ACTION3 moves it right.
    # ACTION4 in the first few steps actually looks like it's shifting the 14x3 block across columns 32, 36, 40, 44.
    # ACTION1 then moves it from row 32 to row 28 and then row 24.
    # ACTION3 moves it right.
    
    # Identify the 'player' object (color 14)
    player_coords = np.argwhere(grid == 14)
    if len(player_coords) == 0:
        return grid
    
    # Player bounding box
    y0, x0 = player_coords.min(axis=0)
    y1, x1 = player_coords.max(axis=0)
    
    # Create a mask of the player shape
    player_mask = np.zeros_like(grid, dtype=bool)
    player_mask[y0:y1+1, x0:x1+1] = True
    
    # Relative coordinates of the player cells
    rel_coords = player_coords - [y0, x0]
    
    # New position candidate
    dy, dx = 0, 0
    if action == 1: # Up
        dy, dx = -1, 0
    elif action == 3: # Right
        dx = 1
    elif action == 4: # Down/Left? No, based on deltas, ACTION4 is "Move Right" in steps of 4 columns.
        # In the observations, ACTION4 moves color 14 from c32 to c36, then c40, then c44.
        # This is a<|channel>thought
        # Let's assume ACTION4 is 'Right' and ACTION1 is 'Up'.
        # The observed ACTION4 shifts by 4 units.
        # dx = 4
        pass
    
    # Special case for ACTION4 as seen in data
    if action == 4:
        # Move right by 4
        new_y0, new_x0 = y0, x0 + 4
        # Check if move is possible (not hitting walls)
        # We need to check if any cell at the same relative positions would hit a wall (color 4).
        # Valid cells are colors 1, 0, 9, 14, 2.
        # Possible movement area is restricted by boundaries or specific colors.
        # Grid borders are 7.
        
        # Boundary check
        if new_x0 < 0 or new_x0 + (x1 - x0) >= grid.shape[1]:
            return grid
        
        # Wall collision check
        collision = False
        for ry, rx in rel_coords:
            if grid[new_y0 + ry, new_x0 + rx] == 4:
                collision = True
                break
        
        if not collision:
            # Perform move
            new_grid = grid.copy()
            # Clear old position
            new_grid[player_mask] = 1 # Background color
            # Set new position
            for ry, rx in rel_coords:
                new_grid[new_y0 + ry, new_x0 + rx] = 14
            return new_grid

    if action == 1:
        # Move up by 4? No, ACTION1 moves it from r32 to r28 then r24. That's a shift of 4.
        new_y0, new_x0 = y0 - 4, x0
        if new_y0 < 0 or new_y0 + (y1 - y0) >= grid.shape[0]:
            return grid
        
        collision = False
        for ry, rx in rel_coords:
            if grid[new_y0 + ry, new_x0 + rx] == 4:
                collision = True
                break
        
        if not collision:
            new_grid = grid.copy()
            new_grid[player_mask] = 1
            for ry, rx in rel_coords:
                new_grid[new_y0 + ry, new_x0 + rx] = 14
            return new_grid

    if action == 3:
        # Right move. In the data, ACTION3 is more complex. It seems to interact with other blocks.
        # Let's simplify and assume it's a standard movement.
        # The delta for ACTION3 shows color 14 moving right and some cells changing to 3? No, that's color 3.
        # Wait, the delta says "r24c44:3x4". That means value 3, count 4.
        # Color 3 isn't in the initial colors list [0, 1, 2, 3, 4, 7, 9, 14].
        # Actually, looking at the INITIAL GRID, there are no 3s.
        # This suggests ACTION3 might be a 'push' or 'interact'.
        pass

    return grid

def is_level_complete(grid):
    # Win state usually involves reaching a target (color 0) or collecting all items (color 9).
    # Or perhaps the player block (color 14) reaches a specific area.
    # Looking at the observed transitions, none of them are marked as win states.
    # But typically, if the player object overlaps the target object (color 0), it's complete.
    player_coords = np.argwhere(grid == 14)
    target_coords = np.argwhere(grid == 0)
    if len(player_coords) == 0 or len(target_coords) == 0:
        return False
    
    # Check for overlap
    for py, px in player_coords:
        for ty, tx in target_coords:
            if py == ty and px == tx:
                return True
    return False