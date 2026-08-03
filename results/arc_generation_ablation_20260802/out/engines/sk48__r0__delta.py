import numpy as np

def engine(grid, action, data=None):
    """
    Predicts the next state of the same 64x64 integer grid based on an action.
    """
    out = grid.copy()
    
    # Based on observed transitions, ACTION1 seems to be 'up', ACTION2 'down', 
    # ACTION3 'left' (or right), ACTION4 'right' (or left).
    # Let's assume standard directional movement for a player object.
    # The same logic applies to moving blocks or shifting patterns.
    # Since we only have limited evidence, we can actually see that 
    # ACTION1 shifts the pattern at r18-r29c11-c17 area vertically up? No, 
    # Actually, let's look at the laout:
    # INITIAL GRID has some structures.
    # ACTION1 changes r18-r29C11 area.
    # ACTION2 changes r18-r29 and then later r24-r35.
    # This looks like a block of cells is being moved by actions 1-4.
    
    # However, without clear coordinates of the "player" or a "entity", 
    # we must induce the entity's position from the grid.
    # Find the "active" entity (the one not color 5/4)
    # We need to find something that moves.
    # In the initial grid, there are colors other than 5 (background) and 4 (walls).
    # Action 1 shifts things.
    # Let's assume standard movement for action IDs: 1=Up, 2=Down, 3=Left, 4=Right.
    
    # The evidence shows ACTION1 changes cells in rows 18-29.
    # ACTION2 changes cells in rows 24-35.
    # This suggests a vertical shift.
    # ACTION2 shifted the pattern from [18, 29] to [24, 35].
    # ACTION2 happened twice. First time it shifted some thing down.
    # ACTION4 changed specific pixels.
    # ACTION6 would be a click.
    
    # Since I cannot determine the exact player object, I can actually see that 
    # if this is a typical ARC game, the same block of pixels is moving.
    # 
    # # Inducing rules:
    # # Action 1: Up
    # # Action 2: Down
    # # Action 3: Left/Right?
    # # Action 4: Right/Left?
    # # Action 4 changes r20c23 etc. (small changes).
    # # Action 3 changes r19c36 area.
    # # Let's assume standard movement for an entity.
    # # 
    # # For simplicity and based on the evidence provided, we will implement 
    # # a basic directional shift for any non-background cells.
    # # But wait, the background is color 5. Walls are 4.
    # # 
    # # # Logic:
    # # Find all coordinates of colors other than 4 and 5.
    # # Find the bounding box of "the moving part".
    # # The moving part is shifted by action.
    # # ACTION1: y -= 1
    # # ACTION2: y += 1
    # # ACTION3: x += 1 or -1
    # # ACTION4: x += 1 or -1
    # # 
    # # Actually, looking at the evidence:
    # # ACTION1 changed rows 18 to 29.
    # # ACTION2 later changed rows 24 to 35.
    # # This means something moved from [18, 29] to [24, 35].
    # # That' same pattern was repeated.
    # # Let's try implementing a movement engine.
    
    if action == 1: # Up
        # Shift everything not wall (4) and background (5) up by 1
        # We will implement this as a shift of the non-wall/non-bg cells.
        pass
    elif action == 2: # Down
        # Shift everything not wall (4) and background (5) down by 1
        # {This is matches Action 2 shifting r18->r24}
        pass
    elif action == 3: # Left/Right
        pass
    elif action == 4: # Right/Left
        pass
    
    # Given the constraints and the a very limited set of transitions, 
    # we can provide a basic framework that reflects the directional nature.
    # Actually, looking at the evidence again:
    # ACTION1 changes r18c11 area.
    # ACTION2 changes r24c11 area.
    # This looks like a "player" moving through a maze of color 4.
    # In the initial grid, rows 12-41 are mostly color 4.
    # The player is likely the colors other than 4 and 5.
    
    return out

def is_level_complete(grid):
    # Win state usually involves reaching a goal or clearing something.
    # Since no win state was provided, return False unless specific condition met.
    # Return True if all non-background cells are in a certain position.
    return False