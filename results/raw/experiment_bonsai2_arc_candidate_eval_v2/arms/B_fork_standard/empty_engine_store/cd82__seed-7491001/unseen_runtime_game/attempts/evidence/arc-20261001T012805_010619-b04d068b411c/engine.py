import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Find player position (color 15)
    player_pos = None
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == 15:
                player_pos = (r, c)
                break
        if player_pos:
            break
    
    if player_pos is None:
        return g
    
    pr, pc = player_pos
    
    # Determine movement direction based on action
    # Action mapping inferred from ARC conventions and observed behavior:
    # ACTION1: Up, ACTION2: Down, ACTION3: Left, ACTION4: Right
    # ACTION5: Special (toggle/interact), ACTION6: Click, ACTION7: ?
    
    dr, dc = 0, 0
    if action == 1:
        dr, dc = -1, 0
    elif action == 2:
        dr, dc = 1, 0
    elif action == 3:
        dr, dc = 0, -1
    elif action == 4:
        dr, dc = 0, 1
    elif action == 5:
        # Toggle/interact with nearby objects
        # From observation: ACTION5 changed color 0 to 15 in a specific region
        # It seems to convert adjacent 0s to 15s or interact with the player's surroundings
        # Looking at the delta for ACTION5: r34c27:15x10 through r38c27:15x10
        # This converted a block of 0s to 15s. The player was likely near this area.
        # Let's check what's around the player and convert 0s to 15s in a pattern
        # Actually, looking more carefully, it seems like ACTION5 might be "collect" or "transform"
        # For now, let's handle it as converting adjacent 0 cells to 15
        pass
    elif action == 6:
        # Click action - from observations, clicking changes bottom-right corner cells
        # Each click seems to change one cell at (63, 63-n) from something to 5
        if data is not None:
            px = data.get('x', 0)
            py = data.get('y', 0)
            # The observed behavior: each ACTION6 changes exactly one cell in row 63
            # from right to left, setting it to 5
            # We need to track state somehow... but engine should be pure.
            # Looking at the sequence: clicks changed c63, then c62, c61, etc.
            # This suggests there's a counter or the grid itself encodes progress.
            # Since we can't maintain external state, let's look for a pattern in the grid.
            # Actually, re-reading: the deltas show r63c63:5x1, then r63c62:5x1, etc.
            # This means the cell that WASN'T 5 becomes 5. Let's find the first non-5 
            # cell from the right in row 63 and set it to 5.
            for c in range(g.shape[1] - 1, -1, -1):
                if g[63, c] != 5:
                    g[63, c] = 5
                    break
        return g
    
    if action == 5:
        # From observation: converted a vertical strip of 0s to 15s
        # The player position before ACTION5 was somewhere near rows 34-38, cols 27-36 area
        # Let's convert all 0 cells adjacent (within some radius) to 15
        # Looking at the delta more carefully: it changed exactly the 0-region to 15
        # The 0 region was at bbox=(34, 27, 43, 36) based on obj13
        # But only rows 34-38 were changed (not all the way to 43)
        # Actually wait - let me re-examine. The initial grid has 0s at r34-r43, c27-c36
        # ACTION5 changed r34c27 through r38c27 (only 5 rows, not 10)
        # This is confusing. Let me just implement movement for now and handle 5 specially.
        
        # Simple approach: find all 0 cells that are "reachable" or adjacent to player
        # and convert them to 15. Or maybe it converts the entire connected component of 0s?
        # For simplicity, let's convert 0s in a small neighborhood around the player
        for dr2 in range(-2, 3):
            for dc2 in range(-2, 3):
                nr, nc = pr + dr2, pc + dc2
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]:
                    if g[nr, nc] == 0:
                        g[nr, nc] = 15
        return g
    
    # Movement logic with gravity/falling
    new_r, new_c = pr + dr, pc + dc
    
    # Check bounds
    if new_r < 0 or new_r >= g.shape[0] or new_c < 0 or new_c >= g.shape[1]:
        return g
    
    target_val = g[new_r, new_c]
    
    # Can move into empty (0) or same-color (15) spaces
    # Cannot move into walls (4, 5, etc.) unless pushing is allowed
    
    if target_val == 0 or target_val == 15:
        # Move player
        g[pr, pc] = 0  # Leave behind... wait, what does the player leave?
        # Looking at deltas, when player moves, the old position becomes 0 (empty)
        # Actually no - looking at ACTION3 delta, it's complex. The player seems to 
        # LEAVE A TRAIL of color 2 (the "path" color).
        
        # Let me re-analyze: In the initial grid, there are color 2 cells forming a border
        # around the central area (rows 24-32, cols 25-38). When the player moves,
        # it seems to create/modify these 2-colored paths.
        
        # This is getting complex. Let me implement basic movement first.
        g[new_r, new_c] = 15
        g[pr, pc] = 0
    else:
        # Blocked - can't move
        return g
    
    return g

def is_level_complete(grid):
    # Check if all 15s have been collected or some win condition is met
    # From observations, the game involves moving a 15-colored object around
    # Win state might be when certain conditions are met
    # For now, check if there are no more 15s remaining (all collected)
    # Or perhaps when specific targets are reached
    count_15 = np.sum(grid == 15)
    # If all 15s are gone, level complete? Or maybe they need to reach specific spots.
    # Without clear win state data, let's assume completion when no 15s remain
    # in "active" areas, or when a specific pattern is achieved.
    # Given limited info, return False by default unless we detect a clear win pattern.
    return count_15 == 0