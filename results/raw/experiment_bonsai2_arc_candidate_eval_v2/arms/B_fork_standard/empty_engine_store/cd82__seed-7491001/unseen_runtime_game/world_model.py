import numpy as np

def engine(grid, action, data):
    g = grid.copy()
    
    # Track a counter at bottom-right corner for click actions
    if action == 6 and data is not None:
        px, py = data['x'], data['y']
        # Check if clicking on specific areas triggers changes
        # From observations, clicks seem to just decrement a counter at r63c63 area
        # The pattern shows r63c63 changing from 4->5, then r63c62 from 4->5, etc.
        # This looks like a timer/counter that fills up row 63 from right to left
        
        # Find the current state of row 63
        # Row 63 starts as all 4s (from initial grid)
        # After each click, one more cell becomes 5, moving leftward
        # Let's find how many cells in row 63 are already 5
        row63 = g[63]
        count_5 = np.sum(row63 == 5)
        
        # If there's space, convert next cell from right
        # Actually looking more carefully:
        # Initial: r63 is all 4s
        # After click 1: r63c63 becomes 5
        # After click 2: r63c62 becomes 5  
        # So it fills from right to left
        
        # Find the rightmost 4 and change it to 5
        for c in range(63, -1, -1):
            if g[63, c] == 4:
                g[63, c] = 5
                break
    
    elif action == 3 or action == 4:
        # These seem to be directional movements that affect a "ball" or object
        # Looking at the patterns, actions 3 and 4 move something diagonally
        # Action 3 appears to move down-left, action 4 moves down-right (or similar)
        
        # The changes show color 2 (border) being replaced by 5 (background) 
        # and color 15 appearing in diagonal lines
        # This looks like a ball bouncing around inside a container
        
        # Let me analyze the structure:
        # There's a diamond/diamond-shaped area with color 2 borders and 15 fill
        # centered around rows 24-37, cols 25-38
        
        # Actually looking more carefully at the deltas:
        # Action 3 creates a pattern where 2s become 5s on one side and 15s appear
        # Action 4 does the opposite
        
        # It seems like there's an object (color 15) moving within a bounded area
        # The bounds are marked by color 2
        
        # For simplicity, let me try to identify the moving object
        # Color 15 objects seem to be the ones moving
        
        # Find all 15-colored cells that might be the "ball"
        # From initial grid, obj12 is color 15 at bbox=(25,26,31,37)
        # And obj9 is color 15 at bbox=(8,3,12,12)
        
        # The movement patterns suggest the ball bounces off walls (color 2)
        
        # This is complex - let me implement basic physics for the 15-ball in the center area
        # The container appears to be bounded by color 2 pixels
        
        # Find the bounding box of the central container (color 2 region around rows 24-37)
        mask_2 = (g == 2)
        if np.any(mask_2):
            rows_with_2 = np.where(mask_2)[0]
            cols_with_2 = np.where(mask_2)[1]
            
            # Focus on the central container (rows ~24-37)
            central_mask = mask_2.copy()
            central_mask[:24] = False
            central_mask[38:] = False
            
            if np.any(central_mask):
                r_min, r_max = np.where(central_mask)[0].min(), np.where(central_mask)[0].max()
                c_min, c_max = np.where(central_mask)[1].min(), np.where(central_mask)[1].max()
                
                # Find the 15-colored object within this region
                mask_15 = (g == 15) & (np.arange(g.shape[0])[:, None] >= r_min) & \
                          (np.arange(g.shape[0])[:, None] <= r_max) & \
                          (np.arange(g.shape[1])[None, :] >= c_min) & \
                          (np.arange(g.shape[1])[None, :] <= c_max)
                
                if np.any(mask_15):
                    ball_rows, ball_cols = np.where(mask_15)
                    ball_r = int(np.mean(ball_rows))
                    ball_c = int(np.mean(ball_cols))
                    
                    # Determine movement direction based on action
                    if action == 3:
                        dr, dc = -1, -1  # up-left? or down-right?
                    else:  # action == 4
                        dr, dc = -1, 1  # up-right? or down-left?
                    
                    # Actually from the patterns, it seems like:
                    # Action 3 moves the ball in one diagonal direction
                    # Action 4 moves it in another
                    
                    # Let me try: action 3 = move left+down, action 4 = move right+down
                    # But the deltas show changes going both up and down...
                    
                    # Simplified approach: just shift the 15-ball by one step
                    new_r = ball_r + dr
                    new_c = ball_c + dc
                    
                    # Check bounds
                    if r_min < new_r < r_max and c_min < new_c < c_max:
                        # Move the ball
                        g[ball_rows, ball_cols] = 5  # clear old position (to background)
                        g[new_r, new_c] = 15  # place at new position
    
    elif action == 5:
        # Action 5 seems to fill certain areas with color 15
        # From observations: fills rows 34-38 cols 27-36 with 15, then rows 39-43 cols 27-31
        # This looks like filling a specific region
        
        # The pattern suggests filling an area that was previously empty (color 0 or 5)
        # Looking at initial grid, there's a 0-colored region at rows 34-43, cols 27-36
        
        # Let me find and fill the largest contiguous non-background region in the lower half
        mask_fillable = (g == 0) | (g == 5)
        
        # Focus on the central-lower area where the 0-region is
        target_region = np.zeros_like(g, dtype=bool)
        target_region[34:44, 27:37] = True
        
        # Fill cells in this region that are 0 with 15
        for r in range(34, min(44, g.shape[0])):
            for c in range(27, min(37, g.shape[1])):
                if g[r, c] == 0:
                    g[r, c] = 15
    
    elif action == 1 or action == 2:
        # Actions 1 and 2 also seem to move objects
        # Action 1 appears similar to action 3 but different direction
        # Action 2 appears similar to action 4 but different direction
        
        # From the deltas, these create diagonal patterns of 2s becoming 5s
        # and 15s appearing/disappearing
        
        # Similar logic to actions 3/4 but potentially different directions
        mask_2 = (g == 2)
        if np.any(mask_2):
            central_mask = mask_2.copy()
            central_mask[:24] = False
            central_mask[38:] = False
            
            if np.any(central_mask):
                r_min, r_max = np.where(central_mask)[0].min(), np.where(central_mask)[0].max()
                c_min, c_max = np.where(central_mask)[1].min(), np.where(central_mask)[1].max()
                
                mask_15 = (g == 15) & (np.arange(g.shape[0])[:, None] >= r_min) & \
                          (np.arange(g.shape[0])[:, None] <= r_max) & \
                          (np.arange(g.shape[1])[None, :] >= c_min) & \
                          (np.arange(g.shape[1])[None, :] <= c_max)
                
                if np.any(mask_15):
                    ball_rows, ball_cols = np.where(mask_15)
                    ball_r = int(np.mean(ball_rows))
                    ball_c = int(np.mean(ball_cols))
                    
                    if action == 1:
                        dr, dc = 1, -1  # down-left
                    else:  # action == 2
                        dr, dc = 1, 1  # down-right
                    
                    new_r = ball_r + dr
                    new_c = ball_c + dc
                    
                    if r_min < new_r < r_max and c_min < new_c < c_max:
                        g[ball_rows, ball_cols] = 5
                        g[new_r, new_c] = 15
    
    return g

def is_level_complete(grid):
    # Check for win condition
    # From the observations, no explicit win state was shown
    # A reasonable guess: all color 4 cells in row 63 have been converted to 5
    # Or some other completion criterion
    
    # Simple heuristic: check if a specific pattern is achieved
    # For now, return False as we don't have clear win state data
    return False