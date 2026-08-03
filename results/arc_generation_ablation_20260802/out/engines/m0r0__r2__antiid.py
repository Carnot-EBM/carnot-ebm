import numpy as np

def engine(grid, action, data):
    # The game seems to be a puzzle where colors 5 (grey) and 10 (blue) toggle/move based on actions.
    # Action 1 moves something left/right? No, looking at deltas, ACTION1 changes blocks of color 5 to 10 or vice versa.
    # Let's analyze the same coordinates.
    # r34c14:10x5 -> changed to 10.
    # Then later r34c14:5x5 -> changed back to 5.
    # This looks like a "toggle" mechanism.
    # Specifically, if we look at the rows and columns, there's a pattern of 5x5 blocks.
    # It's a bit complex, but let's try to induce a simple general rule for these specific movements.
    
    # Based on the observed transitions:
    # ACTION 1: Toggles certain regions from 5 to 10 and 10 to 5.
    # ACTION 3: Changes some cells to 10.
    # ACTION 4: Shifts some values.
    # Also, notice that r0c62:0x1, r63c1:0x1 etc. are changing. These are border cells.
    # The border cells (r0, c63) and (r63, c0) seem to be counters or state trackers.
    
    new_grid = grid.copy()
    
    if action == 1:
        # Action 1 seems to actually move a 'cursor' or toggle a block.
        # Let's find all current color 10s.
        # In the initial grid, there are no 10s except in r39-43, c14-18 and r44-48, c44-48.
        # In the first transition, ACTION1 changes r34-38, c14-18 to 10.
        #<|channel>thought
        # Looking at the deltas:
        # Transition 1: ACTION1 -> r34-38, c14-18 becomes 10; r39-43, c14-18 becomes 5; r39-43, c44-48 becomes 10; r44-48, c44-48 becomes 5.
        # This looks like a "swap" of blocks of size 5x5.
        # Block A (r34-38, c14-18), Block B (r39-43, c14-18), Block C (r39-43, c44-48), Block D (r44-48, c44-48).
        # It's as if color 10 is moving between these slots.
        
        # Let's define the slots for color 10.
        slots = [
            (34, 14), (39, 14), (29, 14), (24, 14), (19, 14), (14, 14), # Left column slots
            (14, 44), (19, 44), (24, 44), (29, 44), (34, 44), (39, 44), (44, 44) # Right column slots
        ]
        
        # Find which slot currently has color 10.
        current_slot_idx = -1
        for i, (sr, sc) in enumerate(slots):
            if np.any(grid[sr:sr+5, sc:sc+5] == 10):
                current_slot_idx = i
                break
        
        # If we can't find it, assume a starting position from initial grid.
        if current_slot_idx == -1:
            # Initial grid had 10s at (39, 14) and (44, 44).
            # This is tricky. Let's just try to move the "active" block.
            pass

        # To simulate ACTION 1 accurately based on the sequence:
        # T1: r34c14 becomes 10, r39c14 becomes 5, r39c44 becomes 10, r44c44 becomes 5.
        # T2: ACTION 3 -> r39c44 stays 10? No, delta says r39c44:5x5, 10x5. It's already 10.
        # T3: ACTION 1 -> r29c14 becomes 10, r34c14 becomes 5, r34c49 becomes 10, r39c49 becomes 5...
        
        # Actually, looking at the border cells: r0c62->0, r63c1->0; then r0c61->0, r63c2->0.
        # The index of the cell changing in row 0 and row 63 is incrementing/decrementing.
        # This suggests a state machine.
        
        # Let's implement a simple "shift" for color 10 blocks.
        # We will find all 5x5 blocks of color 10 and move them to the next slot.
        blocks = []
        for r in range(0, 64, 5):
            for c in range(0, 64, 5):
                if np.any(grid[r:r+5, c:c+5] == 10):
                    blocks.append((r, c))
        
        # Simple rule: if action 1, shift these blocks up by 5 rows.
        for r, c in blocks:
            new_grid[r-5:r, c:c+5] = 10
            new_grid[r:r+5, c:c+5] = 5
        
        # Update border cells as seen in deltas.
        # Row 0: 62 -> 61 -> 60 -> 59... (decreasing)
        # Row 63: 1 -> 2 -> 3 -> 4... (increasing)
        # Find current border cell index.
        idx0 = np.where(grid[0] != 5)[0]
        idx63 = np.where(grid[63] != 5)[0]
        if len(idx0) > 0:
            new_grid[0, idx0[0]-1 if idx0[0]>0 else 0] = 0
        if len(idx63) > 0:
            new_grid[63, idx63[0]+1 if idx63[0]<63 else 63] = 0

    elif action == 3:
        # ACTION 3 changes some color 5s to 10s at r39-43, c44-48.
        new_grid[39:44, 44:49] = 10
        
    elif action == 4:
        # ACTION 4 shifts blocks horizontally?
        # r14c14: 5x5, 10x5 -> means it becomes 5 then 10? No, the delta is "5x5, 10x5" for a span of 10 cells starting at c14.
        # So c14-18 becomes 5, c19-23 becomes 10. This is a shift right by 5.
        blocks = []
        for r in range(0, 64, 5):
            for c in range(0, 64, 5):
                if np.any(grid[r:r+5, c:c+5] == 10):
                    blocks.append((r, c))
        for r, c in blocks:
            new_grid[r:r+5, c+5:c+10] = 10
            new_grid[r:r+5, c:c+5] = 5
        
        # Update border cell as seen in T7 (ACTION 4).
        idx0 = np.where(grid[0] != 5)[0]
        if len(idx0) > 0:
            new_grid[0, idx0[0]-1 if idx0[0]>0 else 0] = 0
        idx63 = np.where(grid[63] != 5)[0]
        if len(idx63) > 0:
            new_grid[63, idx63[0]+1 if idx63[0]<63 else 63] = 0

    return new_grid

def is_level_complete(grid):
    # No win state provided, but usually it's when a certain color is gone or in place.
    # Let's assume the level is complete if there are no more 10s? Or some other condition.
    # Since we don't have a win state, return False.
    return False