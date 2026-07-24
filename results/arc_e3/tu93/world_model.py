import numpy as np

import numpy as np

def engine(grid, action, data):
    H, W = grid.shape
    new_grid = grid.copy()
    
    # Find the player (color 9)
    player_pos = np.argwhere(new_grid == 9)
    if len(player_pos) == 0:
        return new_grid
    
    py, px = player_pos[0]
    
    # Determine movement direction based on action
    # Action 1: Up, 2: Down, 3: Right, 4: Left
    dy, dx = 0, 0
    if action == 1:
        dy, dx = -1, 0
    elif action == 2:
        dy, dx = 1, 0
    elif action == 3:
        dy, dx = 0, 1
    elif action == 4:
        dy, dx = 0, -1
    else:
        # Action 6 is click, others are no-op or handled differently
        # Based on data, only 1-4 seem to move the player in the grid logic
        # Action 6 might be for interaction, but let's assume movement is 1-4
        return new_grid
        
    ny, nx = py + dy, px + dx
    
    # Check bounds
    if 0 <= ny < H and 0 <= nx < W:
        target_val = new_grid[ny, nx]
        
        # Movement logic:
        # 0 (empty): Move into it
        # 2 (wall/obstacle?): Cannot move? Or push?
        # 5 (background): Cannot move?
        # 4 (item?): Collect?
        # 14 (goal?): Win?
        
        # From observations:
        # Player moves into 0 cells.
        # Player seems to push or interact with 4?
        # Let's look at the deltas.
        # Action 4 (Left): Player moved from (16,18) to (16,15)? No, let's trace.
        # Initial player at (16,18) [9x2,4x1 -> 9 at 18,19? No, 9x2 means 2 cells of 9. 
        # Row 16: 5x15, 9x2, 4x1... -> cols 15,16 are 9. Col 17 is 4.
        # Wait, r16: 5x15 (0-14), 9x2 (15-16), 4x1 (17). So player is at 15,16.
        # Action 4 (Left): Delta r16c15:0x3. This clears the player's old position?
        # And r16c21:9x2,4x1. This places player at 21,22? And 4 at 23?
        # This suggests the player moved LEFT from 15,16 to... wait.
        # If player was at 15,16 and moved Left, they should go to 14,15.
        # But the delta shows changes at 15 (clearing) and 21 (placing).
        # This implies the player moved from 15,16 to 21,22? That's a jump.
        # Or maybe the player is a single cell and the 9x2 is a representation artifact?
        # Let's re-read the grid format. "9x2" means two cells of color 9.
        # If the player is 2 cells wide, moving left by 1 would shift the block.
        # But the delta shows the old position (15-17) becoming 0, and a new position (21-23) becoming 9,9,4.
        # This looks like the player moved from col 15-16 to col 21-22. That's a move of +6 columns.
        # Action 4 is Left. This is confusing.
        
        # Let's look at Action 2 (Down).
        # Player at 21-23 (from previous step).
        # Delta: r21c21:0x3, r22c21:0x3, r23c21:0x3. Clears old pos.
        # r27c21:9x3, r28c21:9x3, r29c21:9x1,4x1,9x1. Places new pos at 27-29.
        # Move from row 21 to 27. That's +6 rows.
        # Action 2 is Down. So Down moves +6 rows?
        
        # Let's look at Action 3 (Right).
        # Delta: r63c63:0x1. This is at the bottom right.
        # This doesn't seem to move the player. Maybe it's a counter?
        
        # Let's look at Action 1 (Up).
        # Delta: r21c21:9x1,4x1,9x1... Places player at 21.
        # r27c21:0x3... Clears player at 27.
        # Move from 27 to 21. That's -6 rows.
        # Action 1 is Up. So Up moves -6 rows.
        
        # Hypothesis: The player moves in steps of 6 cells in the direction of the action.
        # The player is 3x3? Or 3x2?
        # Row 16: 9x2, 4x1. Width 3.
        # Row 21-23: 9x3. Height 3.
        # So the player is a 3x3 block, but the center or one cell is different (4)?
        # In row 16, it was 9,9,4. In row 29, it was 9,4,9.
        # It seems the player is a 3x3 block of 9s, with one cell being 4 (maybe a direction indicator or item).
        
        # Let's assume the player is a 3x3 block centered at (py, px).
        # But the grid shows 9s. Let's find the bounding box of 9s.
        
        # Actually, let's just implement the movement as a shift of the 3x3 block.
        # Find all 9s.
        # If action is 1 (Up), shift 9s up by 6.
        # If action is 2 (Down), shift 9s down by 6.
        # If action is 3 (Right), shift 9s right by 6.
        # If action is 4 (Left), shift 9s left by 6.
        
        # But we need to handle the 4 as well. The 4 seems to be part of the player.
        # Let's treat 9 and 4 as the player.
        
        player_mask = (new_grid == 9) | (new_grid == 4)
        player_cells = np.argwhere(player_mask)
        
        if len(player_cells) == 0:
            return new_grid
            
        # Clear old player position
        new_grid[player_mask] = 0
        
        # Shift player cells
        shifted_cells = player_cells + np.array([dy * 6, dx * 6])
        
        # Check bounds and place new player
        for cy, cx in shifted_cells:
            if 0 <= cy < H and 0 <= cx < W:
                # Determine if this cell was 9 or 4
                # We need to know the original value.
                # Let's store the original values.
                pass
        
        # Better approach:
        # Extract the 3x3 block of the player.
        # Find min/max row/col of 9s and 4s.
        rows = player_cells[:, 0]
        cols = player_cells[:, 1]
        min_r, max_r = rows.min(), rows.max()
        min_c, max_c = cols.min(), cols.max()
        
        # The player block is from (min_r, min_c) to (max_r, max_c).
        # It should be 3x3.
        block = new_grid[min_r:max_r+1, min_c:max_c+1].copy()
        
        # Clear the old block
        new_grid[min_r:max_r+1, min_c:max_c+1] = 0
        
        # Calculate new position
        new_min_r = min_r + dy * 6
        new_min_c = min_c + dx * 6
        
        # Check bounds
        if 0 <= new_min_r and new_min_r + 3 <= H and 0 <= new_min_c and new_min_c + 3 <= W:
            # Place the block
            new_grid[new_min_r:new_min_r+3, new_min_c:new_min_c+3] = block
            
    return new_grid

def is_level_complete(grid):
    # Win condition: Player (9) is on the goal (14)?
    # Or all 14s are covered?
    # From the grid, 14 is at (45-47, 57-59).
    # If the player moves there, it might be a win.
    # Let's check if any 9 overlaps with 14.
    # But the player is 9 and 4.
    # If the player is on 14, the 14 might be replaced by 9.
    # So, if there are no 14s left, it's a win?
    # Or if the player is at the position of 14.
    
    # Let's assume the win condition is that the player has reached the 14 cell.
    # Since the player replaces the cell, we can check if the player's position
    # corresponds to where the 14 was.
    # But we don't have the initial grid in the function.
    # We can check if there are no 14s left in the grid.
    
    return np.sum(grid == 14) == 0

import numpy as np

def is_level_complete(grid):
    """
    Returns True if the grid represents a completed level for ARC-AGI-3 task 'tu93'.
    
    The win condition for this task is that the grid contains exactly one contiguous
    region of a single non-background color (the 'target' color), and all other cells
    are the background color (0). The target region must be a solid rectangle or a 
    specific shape defined by the task's logic, but generally, the completion state 
    is characterized by a single, unified object of one color on a black background.
    
    More specifically for 'tu93' (based on common ARC patterns for this ID):
    The task often involves filling or completing a shape. The win state is typically
    a grid where a single color forms a complete, solid rectangular block (or the 
    intended final shape) with no other colors present.
    
    However, without the specific grid examples, the most robust general rule for 
    "completion" in many ARC tasks is:
    1. There is exactly one non-background color present.
    2. All cells of that color form a single connected component (4-way or 8-way).
    3. The background is 0.
    
    Let's refine based on typical 'tu93' behavior if it's a "fill the shape" task:
    The final state is a solid rectangle of one color.
    
    Let's implement a check for:
    - Background is 0.
    - Exactly one other color exists.
    - All pixels of that color form a single connected component.
    """
    grid = np.array(grid)
    
    # Identify background color (usually 0)
    background = 0
    
    # Find all unique colors
    unique_colors = np.unique(grid)
    
    # Remove background from unique colors
    non_bg_colors = [c for c in unique_colors if c != background]
    
    # Must have exactly one non-background color
    if len(non_bg_colors) != 1:
        return False
    
    target_color = non_bg_colors[0]
    
    # Create a binary mask of the target color
    mask = (grid == target_color)
    
    # Check if all target pixels form a single connected component
    # We'll use a simple BFS/DFS to count connected components
    if not np.any(mask):
        return False  # No target pixels
    
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components = 0
    
    for r in range(rows):
        for c in range(cols):
            if mask[r, c] and not visited[r, c]:
                # Start a new component
                components += 1
                if components > 1:
                    return False  # More than one component
                
                # BFS to mark all connected pixels
                queue = [(r, c)]
                visited[r, c] = True
                while queue:
                    curr_r, curr_c = queue.pop(0)
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = curr_r + dr, curr_c + dc
                        if 0 <= nr < rows and 0 <= nc < cols:
                            if mask[nr, nc] and not visited[nr, nc]:
                                visited[nr, nc] = True
                                queue.append((nr, nc))
    
    return True
