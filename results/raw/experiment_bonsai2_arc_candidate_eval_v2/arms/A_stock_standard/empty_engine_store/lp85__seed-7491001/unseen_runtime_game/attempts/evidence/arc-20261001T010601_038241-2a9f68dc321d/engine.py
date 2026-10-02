import numpy as np

def engine(grid, action, data):
    new_grid = grid.copy()
    if action == 6 and data is not None:
        px = data.get('x', 0)
        py = data.get('y', 0)
        
        # Determine which side was clicked (left or right)
        # The board has a left wall at col 0 and right wall around col 56-61
        # Clicking on the left side moves blocks left, clicking on the right side moves blocks right
        # Based on observations: x=4 -> move left, x=58 -> move right
        
        direction = -1 if px < 32 else 1
        
        # Define the "slots" where colored blocks can exist
        # Rows: 19-22, 25-28, 31-34, 37-40, 43-46
        # Cols: 12-15, 18-21, 24-27, 30-33, 36-39, 42-45, 48-51
        slot_rows = [(19, 22), (25, 28), (31, 34), (37, 40), (43, 46)]
        slot_cols = [(12, 15), (18, 21), (24, 27), (30, 33), (36, 39), (42, 45), (48, 51)]
        
        # Collect all block positions and their colors
        blocks = []
        for r_start, r_end in slot_rows:
            for c_start, c_end in slot_cols:
                # Check if this slot has a uniform color that is not the background (4)
                region = new_grid[r_start:r_end+1, c_start:c_end+1]
                unique_colors = np.unique(region)
                if len(unique_colors) == 1 and unique_colors[0] != 4:
                    color = int(unique_colors[0])
                    blocks.append((r_start, c_start, color))
        
        # Sort blocks by row then column to maintain order
        blocks.sort(key=lambda b: (b[0], b[1]))
        
        # Determine which slots are occupied
        occupied_slots = set()
        for r, c, _ in blocks:
            occupied_slots.add((r, c))
        
        # Move blocks in the specified direction
        # Blocks move one slot position in the given direction if the target slot is empty
        
        moved_blocks = {}
        for r, c, color in blocks:
            target_r = r
            target_c = c + direction
            
            # Find the target slot index
            col_idx = None
            for i, (cs, ce) in enumerate(slot_cols):
                if cs <= c <= ce:
                    col_idx = i
                    break
            
            if col_idx is not None:
                new_col_idx = col_idx + direction
                if 0 <= new_col_idx < len(slot_cols):
                    tc_start, tc_end = slot_cols[new_col_idx]
                    target_c = tc_start
                    
                    # Check if target slot is empty
                    if (target_r, target_c) not in occupied_slots and (target_r, target_c) not in moved_blocks:
                        moved_blocks[(target_r, target_c)] = color
                        occupied_slots.discard((r, c))
                        occupied_slots.add((target_r, target_c))
        
        # Apply moves: clear old positions and set new positions
        for r, c, color in blocks:
            if (r, c) not in occupied_slots:
                # This block has moved away
                new_grid[r:r+4, c:c+4] = 4
        
        for (tr, tc), color in moved_blocks.items():
            new_grid[tr:tr+4, tc:tc+4] = color
        
        # Update the left wall indicator (column 0)
        # The wall shows a pattern of 5s that shifts based on movement history
        # Based on observations, clicking left adds 5s to rows below current position
        # Clicking right adds 5s to rows above current position
        # For simplicity, we'll track this as a cumulative effect
        
        # Actually, looking at the data more carefully:
        # When clicking x=58 (right): 5s appear in column 0 at specific rows
        # When clicking x=4 (left): 5s appear in column 0 at different rows
        # The 5s seem to indicate which "lane" or section was last activated
        
        # Let's just handle the block movement for now, as the wall pattern 
        # seems to be a visual indicator rather than affecting gameplay mechanics
        
    return new_grid

def is_level_complete(grid):
    # Check if all colored blocks are in their target positions
    # This is a heuristic - without knowing the exact win condition,
    # we check if there are no movable blocks remaining
    
    slot_rows = [(19, 22), (25, 28), (31, 34), (37, 40), (43, 46)]
    slot_cols = [(12, 15), (18, 21), (24, 27), (30, 33), (36, 39), (42, 45), (48, 51)]
    
    # Count total colored blocks
    block_count = 0
    for r_start, r_end in slot_rows:
        for c_start, c_end in slot_cols:
            region = grid[r_start:r_end+1, c_start:c_end+1]
            unique_colors = np.unique(region)
            if len(unique_colors) == 1 and unique_colors[0] != 4:
                block_count += 1
    
    # A simple completion check: assume level is complete when 
    # blocks reach specific target configurations
    # For now, return False as we don't have explicit win state data
    return False