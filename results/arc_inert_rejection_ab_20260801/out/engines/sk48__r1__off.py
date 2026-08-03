import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where the player controls an object (represented by color 6)
    # and interacts with other elements on the grid.
    # ACTION1-ACTION7 are likely movement or interaction actions.
    # Based on thes transitions, it's a bit complex to induce general rules.
    # However, we looking for specific patterns in the<|channel>thought process.
    # The observed transitions show changes in blocks of colors.
    # laout contains walls (color 5/4), boundaries, and various colored objects.
    # find the 'player' object (color 6).
    
    # Since the provided same-state transitions are not only moving a single pixel,
    # but rather shifting entire blocks of pixels, this a "block-based" world model.
    # Let's identify the player position based on the delta.
    # 
    # Looking at ACTION2 (downward move?), ACTION1 (upward move?).
    # 
    # In ACTION1: r18c11 becomes 6x6, etc. Color 6 is the active agent.
    # In ACTION2: r24c11 becomes 5x2,3x2,5x2... wait, that's color 6 being replaced by something else?
    # No, let's look closer at the run length encoding.
    # r18c11:6x6 means row 18, col 11, value 6, count 6.
    # So color 6 is indeed the agent.
    
    # The game state consists of several regions.
    # There are walls (color 5/4) and potentially interactable objects.
    # Based on the observed transitions, it looks like the agent (color 6) moves in steps.
    # 
    # Let's implement a simple movement engine for the agent (color 6).
    # 
    # Action mapping:
    # ACTION1: Up
    # ACTION2: Down
    # ACTION3: Left
    # ACTION4: Right
    # ACTION6: Click (not used here)
    
    # Find current position of the agent (color 6)
    agent_pos = None
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r, c] == 6:
                agent_pos = (r, c)
                break
        if agent_pos: break

    if agent_pos is None:
        return grid.copy()

    new_grid = grid.copy()
    dr, dc = 0, 0
    if action == 1: dr = -1
    elif action == 2: dr = 1
    elif action == 3: dc = -1
    elif action == 4: dc = 1
    
    # The agent is actually a block of color 6.
    # Let's find all pixels of color 6 and move them.
    agent_pixels = np.argwhere(grid == 6)
    if len(agent_pixels) == 0:
        return grid.copy()
    
    # To avoid leaving trails or overwriting, we need to know what was there.
    # But based on the deltas, it looks like the agent replaces whatever is at the destination.
    # And the previous position is filled with something else (likely color 5 or 4).
    
    # This is tricky because the "background" changes.
    # Let's try to identify the background colors.
    # In ACTION1, r18c11 becomes 6x6. Before that, it was probably not 6.
    # After moving, the old cells are replaced by other values.
    
    # Given the complexity and limited data, let's implement a basic movement for the block of 6s.
    # We assume the agent moves as a unit.
    
    # Find the bounding box of the agent (color 6)
    r_min, c_min = np.min(agent_pixels[:, 0]), np.min(agent_pixels[:, 1])
    r_max, c_max = np.max(agent_pixels[:, 0]), np.max(agent_pixels[:, 1])
    
    # Determine move direction
    dr, dc = 0, 0
    if action == 1: dr = -1
    elif action == 2: dr = 1
    elif action == 3: dc = -1
    elif action == 4: dc = 1
    else: return grid.copy()

    # Move the block
    for r in range(r_min, r_max + 1):
        for c in range(c_min, c_max + 1):
            if grid[r, c] == 6:
                new_grid[r + dr, c + dc] = 6
                # Fill old position with background color? 
                # This is hard to guess. Let's use the value from the destination if we can.
                # Or just a common wall color like 5 or 4.
                # Looking at ACTION2 delta: r24c11 becomes 5x2,3x2,5x2... (color 5)
                # So it seems the agent leaves behind color 5/4.
                new_grid[r, c] = 5 if r < 40 else 4
                
    return new_grid

def is_level_complete(grid):
    # Win state usually involves reaching a target or clearing objects.
    # Without a win state example, we check for a specific condition.
    # Often it's when the agent reaches a certain area.
    # In this game, maybe it's when all of some color are gone.
    # For now, return False as we don't have a clear win condition.
    return False