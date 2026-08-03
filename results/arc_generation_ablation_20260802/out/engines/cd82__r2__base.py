import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a movement/puzzle where certain colors act as walls or obstacles.
    # Based on the transitions, ACTION 2 and 4 are directional movements of an object.
    # Action 2 is likely 'up' or 'left', Action 4 is 'down' or 'right'.
    # Let's assume standard ARC-AGI directions: 0=N, 1=E, 2=S, 3=W (or similar)
    # But based on the same<|channel>thoughtt process, we can actually see that the "object" being moved is color 2.
    # Color 2 is the player character.
    # Movement rules:
    # - It moves in the direction specified by the action.
    # - If it hits a wall (color 4), it stops.
    # - If it hits a void (color 0), it might interact.
    #
    # Looking at the deltas:
    # ACTION 4 (Down/Right?): r21c39:2x1 ... r37c43:2x1. This looks like a path of color 2 moving.
    # ACTION 2 (Up/Left?): r21c39:5x1...
    #
    # Actually, looking closer at the transitions:
    # The object moving is color 2.
    # In ACTION 4, cells are changed to 2.
    # In ACTION 2, cells are changed back to 5 or other colors.
    #
    # Let's refine:
    # Action 2: Move Up
    # Action 4: Move Down
    # Action 6: Click (not used here)
    #
    # However, the provided data is very sparse and complex.
    # A simpler interpretation for this specific level 'cd82':
    # It behaves like a "Snake" or "Path-drawing" game where action moves a head and leaves a trail.
    # Or it's a simple movement puzzle.
    #
    # Given the constraints and the observed delta patterns:
    # Color 2 is the active entity.
    # Action 2: Moves the entity in one direction.
    # Action 4: Moves the entity in another direction.
    #
    # Let's implement a basic movement engine for color 2.
    
    new_grid = grid.copy()
    
    # Find current position of color 2
    pos = np.argwhere(grid == 2)
    if pos.size == 0:
        return new_grid

    # We only handle the last instance of color 2 as the "head"
    head = pos[-1]
    r, c = head
    
    # Map actions to directions (dr, dc)
    # Based on ACTION 4 moving r from 21 to 37 (down) and c from 39 to 43 (right)
    # And ACTION 2 moving r back up and c left.
    directions = {
        2: (-1, 0), # Up
        4: (1, 0),  # Down
        1: (0, -1), # Left
        3: (0, 1),   # Right
    }
    
    if action not in directions:
        return new_grid
        
    dr, dc = directions[action]
    nr, nc = r + dr, c + dc
    
    if 0 <= nr < new_grid.shape[0] and 0 <= nc < new_grid.shape[1]:
        # Simple movement: move the head, leave a trail or just move.
        # The deltas show complex changes, but for a world model we need a general rule.
        # In this specific game, it looks like color 2 is "painting" or "pushing".
        new_grid[nr, nc] = 2
        
    return new_grid

def is_level_complete(grid):
    # Level complete usually means reaching a goal or clearing an area.
    # Looking at the INITIAL grid, there are areas of color 0 and 15.
    # A common win condition is when all cells of a certain target color are gone or replaced.
    # Or if color 2 reaches a specific coordinate.
    # Since no WIN STATE was provided, we'll use a generic completion check.
    # If any cell of color 0 (void) is filled by color 2, maybe that's progress.
    # But without a clear goal, we return False unless a known win state pattern is met.
    return np.sum(grid == 0) == 0