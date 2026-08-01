import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the player controls an object (color 6)
    # and moves it through a maze of walls (color 4).
    # Background color is 5 or 4.
    # Other colors like 8, 9, 14 are targets or goals.
    # ACTION1-7 correspond to keyboard directions (Up, Down, Left, Right)
    # and clicking.
    
    # Mapping actions to movement vectors
    # Based on observed transitions, Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right
    # Action 5: ? Action 6: Click
    
    move_map = {
        1: (-1, 0), # Up
        2: (1, 0),  # Down
        3: (0, -1), # Left
        4: (0, 1),  # Right
    }
    
    if action not in move_map:
        return grid.copy()
    
    dr, dc = move_map[action]
    
    # Find the current position of the agent (color 6)
    # In this case, there' same multiple blocks of color 6.
    # The "agent" block might be a single cell or a complex shape.
    # Let's assume the agent is the set of all cells of color 6.
    # The rest of the game seems to involve moving these blocks of color 6.
    # The uma own logic for the la// This is a specific puzzle where we actually move a group of cells of color 6.
    # The la// This is a<|channel>thought
    # Let's refine the movement logic based on the observed transitions.
    # The agent is the collection of all cells of color 6.
    # The "maze" consists of walls (color 4).
    # Movement is restricted by walls.
    
    # Find all coordinates of color 6
    agent_coords = np.argwhere(grid == 6)
    if len(agent_coords) == 0:
        return grid.copy()
    
    new_grid = grid.copy()
    
    # Check if any part of the agent can move in the direction
    # We need to check if the target positions are not blocked by wall (color 4)
    # and within bounds.
    # try to move every cell of the agent
    can_move = True
    target_coords = []
    for r, c in agent_coords:
        nr, nc = r + dr, c + dc
        if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
            can_move = False
            break
        if grid[nr, nc] == 4: # Wall
            can_move = False
            break
        target_coords.append((nr, nc))
    
    # If movement is possible for the entire group, move them
    if can_move:
        # Clear old position
        for r, c in agent_coords:
            new_grid[r, c] = 5 # Background color
        # Set new position
        for nr, nc in target_coords:
            new_grid[nr, nc] = 6
    else:
        return grid.copy()
    
    return new_grid

def is_level_complete(grid):
    # The level is complete when some condition is met.
    # In many ARC games, the win state is reaching a goal or clearing targets.
    # Let's look at the observed transitions.
    # ACTION4 caused changes to cells of color 3 (color 3 is not seen much).
    # Color 8, 9, 14 are likely goals.
    # Based on common patterns, we might be moving blocks to match a pattern or cover specific colors.
    # Check if all target colors (8, 9, 14) are covered by the agent (color 6).
    targets = [8, 9, 14]
    target_cells = np.argwhere(np.isin(grid, targets))
    if len(target_cells) == 0:
        return False
    
    # If no target cells remain visible, it means they are covered/collected.
    return len(target_cells) == 0