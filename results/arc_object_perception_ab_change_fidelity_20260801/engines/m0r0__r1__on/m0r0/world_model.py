import numpy as np

def engine(grid, action, data):
    # grid: np.ndarray (logical HxW int). Return the predicted next grid (same shape).
    # The game seems to be a puzzle where color 10 blocks move based on actions.
    # Action 1: Move Down? No, looking at the deltas, it's more like shifting blocks.
    # Let's analyze the ACTION1 transitions.
    # r34c14:10x5 -> This means row 34, col 14-18 became color 10.
    # In INITIAL GRID, r34c14 was color 5.
    # Looking at the laout, therethoughts were some color 10 blocks.
    # obj6: color=10 bbox=(39, 14, 43, 18) px=25 centroid=(41.0,16.0)
    # obj7: color=10 bbox=(44, 44, 48, 48) px=25 centroid=(46.0, uma something)
    # These are 5x5 blocks of color 10.
    # Action 1 moves them.
    # Action 3: Changed cells = r39c44:5x5,10x5... this looks like a toggle or merge.
    # Actually, let's look at the action sequence:
    # ACTION1 (level 0->0): changed cells ... r34c14:10x5 ... r39c44:10x5 ...
    # It seems ACTION1 is "Move Up".
    # Let's check if it's moving both blocks.
    # Block 1 (obj6) was at (39, 14). After first ACTION1, it's at (34, 14). Shifted -5 rows.
    # Block 2 (obj7) was at (44, 44). After first ACTION1, it's at (39, 44). Shifted -5 rows.
    # Both shifted up by 5 units.
    # ACTION3:changed cells = r39c44:5x5,10x5. This means row 39-43 col 44-48 became color 5 then 10? No, run length "5x5,10x5" means value 5 for 5 cols, value 10 for 5 cols.
    # Wait, the delta format says "r<row>c<col0>:<v0>x<n0>,<v1>x<n1>".
    # So r39c44:5x5,10x5 means cell (39, 44-48) becomes 5 and (39, 49-53) becomes 10.
    # Let's re-read: "each run walk its <value>x<count> pairs in order... setting <count> consecutive cells starting at the next unfilled column".
    # So r39c44:5x5,10x5 means (39, 44..48)=5, (39, 49..53)=10.
    # But wait, ACTION3 only changes rows 39-43. It looks like it shifted Block 2 from col 44 to col 49. Shifted +5 columns.
    # ACTION1 again: shifts both blocks up by 5 units.
    # Now Block 1 is at (29, 14), Block 2 is at (34, 49).
    # ACTION1 again: shifts both blocks up by 5 units.
    # Now Block 1 is at (24, 14), Block 2 is at (29, 49).
    # ACTION1 again: shifts both blocks up by 5 units.
    # Now Block 1 is at (--- something ---).
    # Action 4: Changed cells = r14c14:5x5,10x5 ... r14c44:10x5,5x5...
    # This moves Block 1 from col 14 to col 19 and Block 2 from col 49 back to col 44.
    # Let's verify the actions:
    # Action 1: Move Up (-5 rows)
    # Action 2: Move Down (+5 rows) - not seen but likely
    # Action 3: Move Right (+5 cols)
    # Action 4: Move Left (-5 cols)
    # The grid also has some "dots" of color 0 moving along the edges.
    # r0c62:0x1 -> r0c61:0x1 -> r0c60:0x1 -> r0c59:0x1. It moves left on row 0.
    # r63c1:0x1 -> r63c2:0x1 -> r63c3:0x1 -> r63c4:0x1. It moves right on row 63.
    # These are just timers or markers.
    
    new_grid = grid.copy()
    
    # Find all 5x5 blocks of color 10
    blocks = []
    for r in range(0, 64-4):
        for c in range(0, 64-4):
            if np.all(grid[r:r+5, c:c+5] == 10):
                blocks.append((r, c))
                # Skip to avoid overlapping detections
                break # this is a bit naive but works for 5x5
    
    # We need to find ALL such blocks. The loop above only finds one per row.
    # Correct way to find 5x5 blocks:
    blocks = []
    visited = np.zeros((64, 64), dtype=bool)
    for r in range(64):
        for c in range(64):
            if grid[r, c] == 10 and not visited[r, c]:
                # Check if it's a 5x5 block
                if r <= 63-4 and c <= 63-4:
                    if np.all(grid[r:r+5, c:c+5] == 10):
                        blocks.append((r, c))
                        visited[r:r+5, c:c+5] = True
                else:
                    # It's just some color 10 pixels, ignore or treat as part of something
                    pass

    # Move the blocks based on action
    dr, dc = 0, 0
    if action == 1: dr = -5
    elif action == 2: dr = 5
    elif action == 3: dc = 5
    elif action == 4: dc = -5
    
    # Update blocks positions
    for r, c in blocks:
        # Clear old position
        new_grid[r:r+5, c:c+5] = 5 # Assuming background is color 5 (or whatever was there)
        # But wait, the background isn't uniform. We need to be careful.
        # The deltas show that when a block moves, it replaces the previous cells with their "original" values?
        # No, looking at ACTION1: "r34c14:10x5". This means row 34 col 14-18 became 10.
        # And what happened to the old cells? "r39c14:5x5" (in later actions).
        # It seems if a block moves, its old position becomes color 5.
        # Let's check the INITIAL GRID again. The area where blocks move is mostly color 5.
        # Actually, look at the laout. Color 11 and 12 are walls.
        # Block 1 moves in the region [9, 58] x [9, 28].
        # Block 2 moves in the region [9, 58] x [28, 53].
        # Both regions are filled with color 5.
        
        new_grid[r+dr : r+dr+5, c+dc : c+dc+5] = 10
        # We must clear the old position first
        # But we need to do this carefully so we don't overwrite other blocks.
    
    # To implement this correctly:
    # 1. Find all blocks of color 10.
    # 2. For each block, calculate new position.
    # 3. Clear all old positions by setting them to color 5.
    # 4. Set all new positions to color 10.
    
    # Re-do movement logic
    blocks = []
    visited = np.zeros((64, 64), dtype=bool)
    for r in range(64):
        for c in range(64):
            if grid[r, c] == 10 and not visited[r, c]:
                if r <= 63-4 and c <= 63-4:
                    if np.all(grid[r:r+5, c:c+5] == 10):
                        blocks.append((r, c))
                        visited[r:r+5, c:c+5] = True
    
    final_grid = grid.copy()
    for r, c in blocks:
        final_grid[r:r+5, c:c+5] = 5
    
    for r, c in blocks:
        nr, nc = r + dr, c + dc
        # Boundary check (simple clamp or ignore)
        if 0 <= nr < 64-4 and 0 <= nc < 64-4:
            final_grid[nr:nr+5, nc:nc+5] = 10
            
    # Handle the "timer" dots of color 0
    # Find dot at row 0
    dot0_col = -1
    for c in range(64):
        if grid[0, c] == 0:
            dot0_col = c
            break
    if dot0_col != -1:
        final_grid[0, dot0_col] = 5
        new_col = dot0_col - 1 if action != 0 else dot0_col # Action 1 moves it left?
        # Looking at ACTION1: r0c62:0x1 -> r0c61:0x1. Yes, any non-zero action moves it.
        # But wait, ACTION3 also moves it? Let's check.
        # The deltas for ACTION3 only show block movement.
        # No, let's re-read: "ACTION3 (level 0->0): changed cells ... r39c44:5x5,10x5".
        # It doesn't mention r0c... or r63c...
        # This means ONLY ACTION1 and ACTION4 move the timer dots?
        # Or maybe they move on every turn?
        # Let's look at the sequence again:
        # ACTION1: r0c62:0x1, r63c1:0x1
        # ACTION3: (no timer change)
        # ACTION1: r0c61:0x1, r63c2:0x1
        # ACTION1: (no timer change) - Wait, ACTION1 (level 0->0) repeated multiple times.
        # Some have timers, some don't. This is confusing.
        # Maybe the same action can be called but not all result in state changes?
        # No, that's unlikely.
        # Let's assume the timer dots move on any action except Action 0 (if it exists).
        # Actually, looking at the deltas:
        # Transition 1 (ACTION1): Timer moves.
        # Transition 2 (ACTION3): Timer does NOT move.
        # Transition 3 (ACTION1): Timer moves.
        # Transition 4 (ACTION1): Timer does NOT move.
        # Transition 5 (ACTION1): Timer does NOT move.
        # Transition 6 (ACTION1): Timer moves.
        # This is very strange. The timer might only move every few turns or based on a specific condition.
        # Given the complexity and the fact that they are just markers, I will ignore them for now as they likely don't affect win condition.

    return final_grid

def is_level_complete(grid):
    # Win state usually involves blocks reaching a target or aligning.
    # In this game, we don't see a win state grid.
    # But typically color 10 blocks would need to reach some goal.
    # Since we have no info, return False.
    return False