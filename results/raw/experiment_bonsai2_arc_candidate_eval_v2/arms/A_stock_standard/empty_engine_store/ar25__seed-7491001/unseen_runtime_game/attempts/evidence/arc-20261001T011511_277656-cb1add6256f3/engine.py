import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track progress on the right wall (column 63)
    # The initial state has color 11 at r0-c63 through r62-c63 and color 5 at r63-c0..c62
    # Progress fills column 63 from top to bottom with color 5
    
    if action in [1, 2, 3, 4]:
        # Movement actions advance the level
        # Find the current progress boundary in column 63
        col63 = g[:, 63]
        
        # Find where the fill stops (transition from 5 to 11)
        # Initially all 11 except row 63 which is part of the bottom bar
        # Actually looking at initial: r0-r62 c63=11, r63 c63=11 (part of 11x1 run)
        # Wait, r63:5x63,11x1 means c0-62=5, c63=11
        
        # After ACTION4: r0c63 becomes 5. So progress starts filling from top.
        # We need to find the first row that is still 11 (not yet filled)
        
        # Count how many rows from top are already 5
        filled_count = 0
        for i in range(64):
            if col63[i] == 5:
                filled_count += 1
            else:
                break
        
        # Advance by one step
        if filled_count < 64:
            g[filled_count, 63] = 5
        
        # Move the objects based on action direction
        # Action 1: Up, Action 2: Down, Action 3: Left, Action 4: Right
        
        # The movable objects appear to be the colored blocks (5 and 4) 
        # with their internal 0 patterns. They move as a group.
        
        # Let's identify the object positions by finding connected components of non-background colors
        # Background is 9 (main area), 10 (wall), 11 (right wall/bottom bar edge)
        
        # Actually, looking more carefully at the transitions:
        # ACTION4 (Right): Objects shift right
        # ACTION1 (Up): Objects shift up  
        # ACTION2 (Down): Objects shift down
        # ACTION3 (Left): Objects shift left
        
        # The objects seem to be the clusters of 5s and 4s with 0s inside them
        # Let me track them differently - find all cells that are part of "objects"
        
        # Object cells are those with color 5 or 4 or 0 that are NOT in the background regions
        # But 0 only appears inside objects, so let's use that
        
        # Find bounding boxes of object groups
        # An object cell is one where grid value is in [0, 4, 5] AND it's not part of the static structure
        
        # Static structure: rows 0-62 have pattern 9x30,10x3,9x30,11x1
        # Row 63 has 5x63,11x1
        # The 10 column (c30-32) and 11 column (c63) are walls
        # The bottom row 63 c0-62 is a floor
        
        # So movable objects are in the region r0-r62, c0-c29 and c33-c62
        # excluding the wall at c30-32
        
        # Let me just shift the entire content based on direction
        # but only for cells that are "object" cells
        
        # Define object mask: cells with value in {0, 4, 5} that are within the play area
        # Play area excludes: c30-32 (wall), c63 (right wall), r63 (bottom bar)
        
        h, w = g.shape
        obj_mask = np.zeros((h, w), dtype=bool)
        
        # Mark potential object cells
        for r in range(h):
            for c in range(w):
                if g[r, c] in [0, 4, 5]:
                    # Exclude static structures
                    if c == 63:  # right wall
                        continue
                    if r == 63:  # bottom bar
                        continue
                    if 30 <= c <= 32:  # middle wall
                        continue
                    obj_mask[r, c] = True
        
        # Now apply movement
        new_g = g.copy()
        
        if action == 1:  # Up
            # Shift objects up by 1
            for r in range(1, h):
                for c in range(w):
                    if obj_mask[r, c]:
                        new_g[r-1, c] = g[r, c]
            # Clear old positions
            for r in range(h):
                for c in range(w):
                    if obj_mask[r, c]:
                        # What should the cleared cell be? Background is 9
                        new_g[r, c] = 9
            # But we need to handle overlaps - process from top to bottom
            # Actually let's do it properly: shift all object cells up by 1
            new_g = g.copy()
            # First clear all object positions
            for r in range(h):
                for c in range(w):
                    if obj_mask[r, c]:
                        new_g[r, c] = 9
            # Then place them shifted up
            for r in range(1, h):
                for c in range(w):
                    if obj_mask[r, c]:
                        target_r = r - 1
                        # Check bounds and collision with walls
                        if target_r >= 0 and not (30 <= c <= 32) and c != 63 and target_r != 63:
                            new_g[target_r, c] = g[r, c]
        
        elif action == 2:  # Down
            new_g = g.copy()
            for r in range(h):
                for c in range(w):
                    if obj_mask[r, c]:
                        new_g[r, c] = 9
            for r in range(h-1):
                for c in range(w):
                    if obj_mask[r, c]:
                        target_r = r + 1
                        if target_r < h and not (30 <= c <= 32) and c != 63 and target_r != 63:
                            new_g[target_r, c] = g[r, c]
        
        elif action == 3:  # Left
            new_g = g.copy()
            for r in range(h):
                for c in range(w):
                    if obj_mask[r, c]:
                        new_g[r, c] = 9
            for r in range(h):
                for c in range(1, w):
                    if obj_mask[r, c]:
                        target_c = c - 1
                        if target_c >= 0 and not (30 <= target_c <= 32) and target_c != 63 and r != 63:
                            new_g[r, target_c] = g[r, c]
        
        elif action == 4:  # Right
            new_g = g.copy()
            for r in range(h):
                for c in range(w):
                    if obj_mask[r, c]:
                        new_g[r, c] = 9
            for r in range(h):
                for c in range(w-1):
                    if obj_mask[r, c]:
                        target_c = c + 1
                        if target_c < w and not (30 <= target_c <= 32) and target_c != 63 and r != 63:
                            new_g[r, target_c] = g[r, c]
        
        g = new_g
    
    elif action == 5:
        # Action 5 seems to just advance progress without moving objects
        col63 = g[:, 63]
        filled_count = 0
        for i in range(64):
            if col63[i] == 5:
                filled_count += 1
            else:
                break
        if filled_count < 64:
            g[filled_count, 63] = 5
    
    elif action == 6:
        # Click - no change observed
        pass
    
    return g

def is_level_complete(grid):
    # Level complete when column 63 is fully filled with 5s (all 64 rows)
    col63 = grid[:, 63]
    return np.all(col63 == 5)