import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the player controls an object (represented by color 6)
    # and interacts with objects of other colors.
    # Based on the observed transitions, ACTION1-ACTION4 are directional movements or interactions.
    # ACTION6 is a click.
    # grid shape is 64x64.
    
    # Identify the current position of the "player" character (color 6).
    # Find all cells of color 6.
    # find the y, x coordinates of thes characters.
    coords = np.argwhere(grid == 6)
    if len(coords) == 0:
        return grid
    
    # We assume there's only one primary player character or a group moving together.
    # Let's use the first one found.
    y, x = coords[0]
    
    # Define movement vectors for actions.
    # Action 1: Up? Action 2: Down? Action 3: Left? Action 4: Right?
    # Looking at the delta changes in ACTION1, r18c11 becomes 6x6, etc.
    # The initial grid has color 6 starting around row 24, col 11.
    # In ACTION1, it moves to row 18. This suggests ACTION1 is UP.
    # ACTION2 moves it from row 18 back down towards row 24, then later from row 24 to row 30.
    # ACTION2 is DOWN.
    # ACTION3 is not clearly a-priori but let's look at the observed transitions.
    # ACTION3 changed cells at r19c36, which is far right of c11.
    # ACTION3 seems to be an interaction that affects other objects.
    # ACTION4 changes cells at r20c23, r20c29, r20c35... moving horizontally.
    # # Let's<|channel>thought
    # Actually, looking closer at the data:
    # ACTION1: (r24, c11) -> (r18, c11). Up.
    # ACTION2: (r18, c11) -> (r24, c11), then (r24, c11) -> (r30, c11). Down.
    # ACTION3: Affects color 8 and 4 in columns 36-42.
    # ACTION4: Affects colors 2, 1, etc. in rows 20-21.
    
    # This looks like a game where you move a block of color 6 and it pushes or interacts with others.
    # The grid contains walls (color 5 and 4).
    # The same pattern repeats in the bottom half (rows 56-60).
    
    # Based on the provided transitions, we can't induce a perfect general rule for all levels.
    # But we can actually see that the player character (color 6) is a 6x6 block.
    # The observed deltas are very specific to these coordinates.
    # la// Let's implement a simple movement model for the "player" block.
    
    new_grid = grid.copy()
    
    # Find the top-left corner of the 6x6 block of color 6.
    coords = np.argwhere(grid == 6)
    if len(coords) == 0:
        return new_grid
    
    y_min, x_min = np.min(coords, axis=0)
    y_max, x_max = np.max(coords, axis=0)
    
    # Movement vectors
    dy, dx = 0, 0
    if action == 1: # UP
        dy = -6
    elif action == 2: # DOWN
        dy = 6
    elif action == 3: # LEFT? No, ACTION3 in data was something else.
        dx = -6
    elif action == 4: # RIGHT?
        dx = 6
    
    # Apply movement if it doesn't hit a wall (color 5).
    # Check boundaries and walls.
    target_y_min, target_x_min = y_min + dy, x_min + dx
    target_y_max, target_x_max = y_max + dy, x_max + dx
    
    if target_y_min < 0 or target_y_max >= 64 or target_x_min < 0 or target_x_max >= 64:
        return new_grid
    
    # Simple collision check with color 5.
    if np.any(grid[target_y_min:target_y_max+1, target_x_min:target_x_max+1] == 5):
        return new_grid

    # Clear old position
    new_grid[y_min:y_max+1, x_min:x_max+1] = grid[y_min:y_max+1, x_min:x_max+1].copy() # This is wrong
    # We need to know what was behind the block. In this game, it seems like the background is color 4 or 5.
    # Let's assume the movement replaces cells with a default "empty" color (color 4).
    
    # To be more accurate, we should just apply the deltas if they match own patterns.
    # But since we must provide an engine, let's use the observed movements.
    
    # For ACTION3 and ACTION4, they seem to trigger events in other parts of the map.
    # The data shows them changing colors at specific coordinates regardless of player pos?
    # No, probably based on alignment.
    
    # Since we can't fully induce the logic from such limited transitions, 
    # we will implement the basic movement for Action 1 & 2 as seen.
    
    if action == 1:
        # Move block of 6s up by 6 units.
        new_grid[y_min:y_max+1, x_min:x_max+1] = 4 # Fill old with 4
        new_grid[target_y_min:target_y_max+1, target_x_min:target_x_max+1] = 6
    elif action == 2:
        # Move block of 6s down by 6 units.
        new_grid[y_min:y_max+1, x_min:x_max+1] = 4
        new_grid[target_y_min:target_y_max+1, target_x_min:target_x_max+1] = 6
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves moving an object to a goal or clearing something.
    # In this game, it's not explicitly shown in the transitions (all are level 0->0).
    # We will return False unless we see a specific win condition.
    return False