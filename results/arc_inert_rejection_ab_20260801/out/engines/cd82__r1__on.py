import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The observed transitions suggest a complex movement of objects/colors.
    # Based on the same-game-id 'cd82' and typical ARC patterns, 
    # we assume this is a puzzle where an object moves or changes state based on actions.
    # ACTION 2, 4 are directional movements.
    # ACTION 6 is click.
    # ACTION 0-7 are keyboard keys.
    
    # Since the exact logic for the same game id 'cd82' in other contexts might not    
    # often be provided, we need to implement a general rule that matches the observations.
    # In thes specific case, the laout contains colors like 0, 2, 3, 4, 5, 15.
    # 5 is background.
    # 4 is wall/boundary.
    # 15 is target/object.
    # 15 can move through some areas.
    # <|channel>thought
    # Action 2: Move Down? Or Right?
    # Action 4: Move Left?
    # Let's look at the deltas.
    # ACTION 4 (level 0->0): r21c39:2x1 ... r37c43:2x1. This looks like it shifted something.
    # ACTION 2 (level 0->0): r21c39:5x1 ... r45c38:2x9.
    # The transitions show color 15 moving and changing positions of color 2.
    
    # Given the constraints and the limited data, we will implement a simplified version
    # where action 2 moves an object down-right and action 4 moves it left.
    # However, without a clear rule, we must be careful.
    # Looking closely at the delta for ACTION 4: cells are changed to 2, 15, 5.
    # It seems like a "snake" or "block" of color 15 is being moved.
    
    # For this specific task, since I cannot deduce the full physics engine from 3 frames,
    # I will provide a skeleton that handles the most likely scenario: movement of blocks.
    
    # But wait, looking at the deltas again:
    # Action 4: changes rows 21 to 37.
    # Action 2: changes rows 21 to 45.
    # This looks like gravity or sliding.
    
    # Let's try to find the coordinates of color 15 before and after.
    # Initial grid has 15s at r8c3, r9c3... (a block) and r25c26... (another block).
    # After Action 4, new 15s appear around r23c38, r24c38, etc.
    # The object at r25-r31 c26-37 shifted right/down.
    
    # Since we must return an executable world model, let's implement a basic shift logic for these actions.
    
    # ACTION 2: Down/Right shift
    # ACTION 4: Left/Up shift
    
    # To be honest, without more data, this is guessing. But I will implement a rule where
    # action 2 shifts all non-background cells down by 1 if possible, and action 4 shifts them left.
    
    # However, looking at the deltas again, it's very specific. It' same as "Sokoban" or "Pushing".
    # Let's assume:
    # Action 2 -> Move Right
    # Action 4 -> Move Left
    # Action 1 -> Move Up
    # Action 3 -> Move Down
    
    # Let's refine based on delta:
    # Action 4 (level 0->0): r21c39:2x1... This looks like something moved to the RIGHT.
    # Wait, ACTION 4 usually means 'Left' in some ARC games, but here it seems to move things right?
    # Actually, look at r21c39:2x1. Before that was background 5. Now it's 2.
    # The block of 15s shifted from c26-37 to c38+. That is a shift to the RIGHT.
    # So Action 4 = Right.
    # Then Action 2 = Down.
    
    # Let's implement basic movement for color 15 and 2.
    
    new_grid = grid.copy()
    if action == 4: # Shift Right
        # Find all non-background cells
        mask = (grid != 5)
        # Simple shift right logic: if cell is not 5 and cell to its right is 5, move it.
        # To avoid overlapping, we iterate backwards.
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]-2, -1, -1):
                if grid[r, c] != 5 and grid[r, c+1] == 5:
                    new_grid[r, c+1] = grid[r, c]
                    new_grid[r, c] = 5
    elif action == 2: # Shift Down
        mask = (grid != 5)
        for r in range(grid.shape[0]-2, -1, -1):
            for c in range(grid.shape[1]):
                if grid[r, c] != 5 and grid[r+1, c] == 5:
                    new_grid[r+1, c] = grid[r, c]
                    new_grid[r, c] = 5
    
    return new_grid

def is_level_complete(grid):
    # Win state usually means color 15 is at a target or some condition is met.
    # In the provided data, there's no win state shown.
    # We'll assume it's complete if all 15s are gone or in a specific area.
    # For now, return False as we don't have a win example.
    return False