import numpy as np

def engine(grid, action, data):
    # The game state involves a grid where certain colors represent different entities.
    # Based on the observed transitions, it's a movement/collection game.
    # Action 1 moves an entity (likely color 10) and changes cells.
    # Action 4 moves another entity or shifts the same entity.
    #<|channel>thought
    # Looking at the deltas, ACTION1 seems to shift blocks of color 10 into positions of color 5.
    # ACTION3 might be a toggle or specific interaction.
    # ACTION4 shifts blocks in the opposite direction.
    # Let's implement a simple rule based on the patterns seen in the deltas.
    
    new_grid = grid.copy()
    
    if action == 1:
        # Shift color 10 blocks from right to left or vice versa?
        # No, looking at the deltas: r34c14:10x5 means row 34, col 14 becomes color 10 for 5 cols.
        # It looks like "filling" areas of color 5 with color 10.
        # We need to find the current 'cursor' or active area.
        # The delta shows that when ACTION1 is pressed, certain regions change to 10.
        # Then later they change back to 5.
        # This suggests a moving window of color 10.
        
        # Since we don't have the full state transition logic (like player position),
        # and the game is likely deterministic based on the grid,
        # let's look for the existing color 10 blocks and move them.
        
        # Find all blocks of color 10
        coords = np.argwhere(grid == 10)
        if coords.size == 0:
            return new_grid
        
        # In this specific level, it seems color 10 moves in a pattern.
        # Let's try to simulate the shift observed in the deltas.
        # Action 1 shifts the block "up" or "left".
        # Action 4 shifts the block "down" or "right".
        
        # Based on the deltas:
        # Transition 1: r34-38 c14 becomes 10, r39-43 c44 becomes 10.
        # Transition 2: ACTION3 changes r39-43 c44 from 10 to 5? No, delta says 5x5, 10x5.
        # It looks like color 10 is moving across the board.
        
        # Given the constraints and the limited data, we will implement a logic that
        # mimics the movement of the color 10 blocks based on the action.
        
        # For simplicity, let's assume ACTION1 moves the 'active' block (color 10)
        # to the next predefined region of color 5.
        
        # This is very hard without knowing the exact sequence.
        # However, looking at the coordinates:
        # ACTION1: (34-38, 14), then (29-33, 14), then (24-28, 14), then (19-23, 14), then (14-18, 14).
        # Then it shifts to col 49: (14-18, 49), etc.
        # It's moving UP in column 14, then shifting to column 49 and moving DOWN?
        # Or maybe it's just cycling through regions.
        
        # Let's try to find the current center of mass of color 10 and move it.
        center = np.mean(coords, axis=0)
        if action == 1:
            # Move "up" or "left"
            # In this game, "up" means decreasing row index.
            new_grid[np.clip(center[0]-5, 0, 63).astype(int):np.clip(center[0]+1, 0, 63).astype(int), 
                      np.clip(center[1]-2, 0, 63).astype(int):np.clip(center[1]+3, 0, 63).astype(int)] = 10
            # Clear old
            new_grid[coords[:,0], coords[:,1]] = 5
    
    return new_grid

def is_level_complete(grid):
    # Win state usually involves clearing all blocks of a certain color or reaching a goal.
    # In this case, we don't have a win state grid, but typically it's when color 10 is gone
    # or in a specific place.
    # Let's assume the level is complete if there are no more blocks of color 10 (or they reach a target).
    # Since we can't be sure, let's check for a common win condition: any cell being 0?
    # The INITIAL GRID has r0c63=0 and r63c0=0. These might be markers.
    # Looking at ACTION1 deltas: r0c62 becomes 0, then r63c1 becomes 0...
    # This looks like a progress bar! The zeros are moving from corners towards each other.
    # Level complete when the zeros meet or cover the board.
    
    zeros = np.argwhere(grid == 0)
    if zeros.size >= 64: # Arbitrary threshold based on the "progress" observed
        return True
    return False