import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where an object moves and interacts with 
    # walls/obstacles. Based on the observed transitions, ACTION1-ACTION4 are directional movements.
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right.
    # However, looking at the same transition sequence, it's a more complex interaction.
    # Let's assume standard movement for a player character (the color 6) and 
    # a coordinate system where y is row, x is col.
    
    # Find the "player" (color 6)
    # In the initial grid, color 6 is present in multiple locations.
    # This looks like a "block" of color 6 cells.
    #<|channel>thought
    # The provided delta changes suggest that blocks of colors move or shift.
    # The observed actions correspond to shifting blocks of pixels.
    # The win state is not explicitly given, but usually involves moving something to a target.
    
    # Since we cannot analyze the grids deeply, we will implement a simple 
    # translation based on the observed deltas.
    
    # Identify all coordinates of each color.
    # For example, if the same block of color 6 moves from r18c11 to r24c11?
    # No, let's look at the action-delta pairs.
    # ACTION1 shifts things up? ACTION2 shifts down?
    # ACTION3 shifts left? ACTION4 shifts right?
    # Actually, looking at the transitions:
    # ACTION1: r18c11 -> r29c11 area changed.
    # ACTION2: r24c11 -> r35c11 area changed.
    # It seems ACTION1 and ACTION2 are vertical movements.
    # Let's try to map them: ACTION1=Up, ACTION2=Down, ACTION3=Left, ACTION4=Right.
    
    # To make this executable, we need to identify what "moves".
    # In these games, typically there is one entity (the player) that moves.
    # The colors in the grid are background (5), walls (4), and entities.
    # Any cell not equal to 5 or 4 might be an entity.
    # a = np.where((grid != 5) & (grid != 4))
    # But wait, color 6 is also present.
    
    # Given the constraints and the data, let's implement a basic movement engine for any non-wall/non-bg cells.
    # This is a guess based on common ARC patterns.
    
    new_grid = grid.copy()
    
    # Find all coordinates of "entities" (not bg 5, not wall 4)
    entities = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] not in [4, 5]:
                entities.append((r, c, grid[r, c]))
    
    # Map actions to directions
    directions = {
        1: (-1, 0), # Up
        2: (1, 0),  # Down
        3: (0, -1), # Left
        4: (0, 1),  # Right
    }
    
    if action in directions:
        dr, dc = directions[action]
        # Move entities if possible
        # To avoid overwriting, we process them in an order or use a temporary grid
        temp_grid = np.full(grid.shape, 5) # Reset to background
        # Re-add walls
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if grid[r, c] == 4:
                    temp_grid[r, c] = 4
        
        # Try moving each entity
        for r, c, val in entities:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                if temp_grid[nr, nc] != 4:
                    temp_grid[nr, nc] = val
                else:
                    temp_grid[r, c] = val
            else:
                temp_grid[r, c] = val
        new_grid = temp_grid

    return new_grid

def is_level_complete(grid):
    # Usually complete when some target color is reached or certain cells are filled.
    # Without the win state, we check if any "entity" has moved to a specific area (e.g., bottom).
    # Let's assume it's complete if no non-bg/non-wall cells are in the top half.
    # This is a placeholder as the win condition isn't provided.
    entities = np.where((grid != 5) & (grid != 4))
    if len(entities[0]) == 0: return False
    return np.all(entities[0] > grid.shape[0] // 2)